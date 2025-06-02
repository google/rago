"""
The main python script for RAG performance analysis and optimization (except iterative retrievals).
"""

import copy
import gc
import numpy as np
import pandas as pd
import time

from itertools import product
from typing import Any, Dict

from config import RetrievalPolicy, EncodeDBPolicy, PhysicalMapping
from utils import get_filtered_df, is_power_of_two, get_power_of_two_list, get_pareto_df


class RAGSweep:

    def __init__(
        self,
        retrieval_policy: RetrievalPolicy,
        encode_db_policy: EncodeDBPolicy,
        stages: list[str] = ["retrieval", "prefill", "decode"],
        sweep_df: dict[str, pd.DataFrame | None] = {
            "rewrite_prefill": None,
            "rewrite_decode": None,
            "encode": None,
            "retrieval": None,
            "rerank": None,
            "prefill": None,
            "decode": None,
        },
        seq_len_inference_prefill: int = 512,
        dec_steps: int = 256,
        num_chips_per_server: int = 4,
    ):
        # Raw performance sweeps
        self.stages = stages
        self.sweep_df = sweep_df

        # Policies on retrieval, db encoding, and integration
        self.retrieval_policy = retrieval_policy

        # Inference parameter constraints
        self.seq_len_inference_prefill = seq_len_inference_prefill
        self.dec_steps = dec_steps

        # Physical constant constraint
        self.num_chips_per_server = num_chips_per_server

        # Get filtered df per stage
        self.sweep_df["prefill"] = get_filtered_df(
            self.sweep_df["prefill"],
            {"seq_len_inference_prefill": self.seq_len_inference_prefill},
        )
        assert len(self.sweep_df["prefill"]) > 0

        self.sweep_df["decode"] = get_filtered_df(
            self.sweep_df["decode"],
            {
                "seq_len_inference_prefill": self.seq_len_inference_prefill,
                "dec_steps": self.dec_steps,
            },
        )
        assert len(self.sweep_df["decode"]) > 0

        self.encode_db_policy = encode_db_policy
        if encode_db_policy.run_encode_db:
            assert "encode" in self.stages
            self.encode_seq_len_inference_prefill = (
                encode_db_policy.encode_seq_len_inference_prefill
            )
            self.encode_db_nvec = encode_db_policy.encode_db_nvec
            assert self.encode_seq_len_inference_prefill is not None
            self.sweep_df["encode"] = get_filtered_df(
                self.sweep_df["encode"],
                {"seq_len_inference_prefill": self.encode_seq_len_inference_prefill},
            )
        else:
            if "encode" in self.stages:
                self.stages.remove("encode")

        if retrieval_policy.run_retrieval:
            assert "retrieval" in self.stages
            self.retrieval_interval = retrieval_policy.retrieval_interval
            assert len(self.sweep_df["retrieval"]) > 0
        else:
            if "retrieval" in self.stages:
                self.stages.remove("retrieval")

        if retrieval_policy.run_rewrite:
            assert "rewrite_prefill" in self.stages
            assert "rewrite_decode" in self.stages
            self.rewrite_seq_len_inference_prefill = (
                retrieval_policy.rewrite_seq_len_inference_prefill
            )
            self.rewrite_seq_len_prefill_with_template = (
                retrieval_policy.rewrite_seq_len_prefill_with_template
            )
            self.rewrite_dec_steps = retrieval_policy.rewrite_dec_steps
            assert self.rewrite_seq_len_inference_prefill is not None
            assert self.rewrite_seq_len_prefill_with_template is not None
            assert self.rewrite_dec_steps is not None
            self.sweep_df["rewrite_prefill"] = get_filtered_df(
                self.sweep_df["rewrite_prefill"],
                {"seq_len_inference_prefill": self.rewrite_seq_len_inference_prefill},
            )
            self.sweep_df["rewrite_decode"] = get_filtered_df(
                self.sweep_df["rewrite_decode"],
                {
                    "seq_len_inference_prefill": self.rewrite_seq_len_prefill_with_template,
                    "dec_steps": self.rewrite_dec_steps,
                },
            )
        else:
            if "rewrite_prefill" in self.stages:
                self.stages.remove("rewrite_prefill")
            if "rewrite_decode" in self.stages:
                self.stages.remove("rewrite_decode")

        if retrieval_policy.run_rerank:
            assert "rerank" in self.stages
            self.rerank_seq_len_inference_prefill = (
                retrieval_policy.rerank_seq_len_inference_prefill
            )
            self.rerank_topk = retrieval_policy.rerank_topk
            assert self.rerank_seq_len_inference_prefill is not None
            assert self.rerank_topk is not None
            self.sweep_df["rerank"] = get_filtered_df(
                self.sweep_df["rerank"],
                {"seq_len_inference_prefill": self.rerank_seq_len_inference_prefill},
            )
        else:
            if "rerank" in self.stages:
                self.stages.remove("rerank")

        assert "prefill" in self.stages and "decode" in self.stages
        assert self.stages[-1] == "decode"

        for stage in self.stages:
            assert (
                self.sweep_df[stage] is not None and len(self.sweep_df[stage]) > 0
            ), f"Stage {stage} has no data."

        print("Effective stages: ", self.stages)

        # Performance cache
        # Key: [stage_name][num_chips or number of servers],
        # Value: latency; possible batch size; performance pareto
        self.performance_pareto_dict = {}
        self.possible_batch_sizes_dict = {}
        self.latency_lookup_table_dict = {}
        for stage in self.stages:
            self.performance_pareto_dict[stage] = {}
            self.possible_batch_sizes_dict[stage] = {}
            self.latency_lookup_table_dict[stage] = {}

        self.all_placement_strategies = None

    """ Helper Functions Starts """

    def scale_up_performance_to_more_chips(
        self,
        stage,
        max_num_chips: int | None = None,
        max_num_retrieval_servers: int | None = None,
    ):
        """
        The existing performance df may stop at certain number of chips. This function
        helps to scale the performance to more chips to support larger-scale sweeps with many chips.


        """
        if stage == "retrieval":
            # This function iterates through the unique num_servers values in the retrieval data,
            # starting from the smallest. For each value, it checks if doubling that number (e.g., 1 → 2, 2 → 4, etc.)
            # exists in the data. If not, continue double the perf until exceeds the max_num_retrieval_servers.

            assert max_num_retrieval_servers is not None

            while max_num_retrieval_servers > max(
                self.sweep_df[stage]["num_servers"].unique()
            ):
                # Get the current retrieval data
                df_retrieval = self.sweep_df[stage]
                # Get the list of unique number of retrieval servers in the df
                num_retrieval_servers_in_df = sorted(
                    df_retrieval["num_servers"].unique()
                )

                # Loop through the sorted retrieval servers
                for num_servers in num_retrieval_servers_in_df:
                    # Check if the double of this num_servers is in the df
                    doubled_num_servers = num_servers * 2

                    # If the doubled num_servers is not in the df and it is less than or equal to max_num_retrieval_servers
                    if doubled_num_servers not in num_retrieval_servers_in_df:
                        # Select rows with the current num_servers
                        num_servers_rows = df_retrieval[
                            df_retrieval["num_servers"] == num_servers
                        ]

                        # Scale up these rows by doubling the values
                        new_data = num_servers_rows.copy()
                        new_data["num_servers"] = doubled_num_servers
                        new_data["num_chips"] *= 2
                        new_data["qps"] *= 2
                        new_data["batch_size"] *= 2

                        # Append the scaled data to the DataFrame
                        df_retrieval = pd.concat(
                            [df_retrieval, new_data], ignore_index=True
                        )

                # Update the performance dictionary for retrieval with the modified DataFrame
                self.sweep_df[stage] = df_retrieval

        else:
            assert max_num_chips is not None
            # This function iterates through the unique num_chips values in the inference data,
            # starting from the smallest. For each value, it checks if doubling that number (e.g., 1 → 2, 2 → 4, etc.)
            # exists in the data. If not, it continues doubling the performance until it exceeds the max_num_chips.

            while max_num_chips > max(self.sweep_df[stage]["num_chips"].unique()):
                # Get the current inference data
                df_inference = self.sweep_df[stage]
                # Get the list of unique number of chips in the df
                num_chips_in_df = sorted(df_inference["num_chips"].unique())

                # Loop through the sorted number of chips
                for num_chips in num_chips_in_df:
                    # Check if the double of this num_chips is in the df
                    doubled_num_chips = num_chips * 2

                    # If the doubled num_chips is not in the df and it is less than or equal to max_num_chips
                    if doubled_num_chips not in num_chips_in_df:
                        # Select rows with the current num_chips
                        num_chips_rows = df_inference[
                            df_inference["num_chips"] == num_chips
                        ]

                        # Scale up these rows by doubling the values
                        new_data = num_chips_rows.copy()
                        new_data["num_chips"] = doubled_num_chips
                        new_data["qps"] *= 2
                        new_data["batch_size"] *= 2

                        # Append the scaled data to the DataFrame
                        df_inference = pd.concat(
                            [df_inference, new_data], ignore_index=True
                        )

                # Update the performance dictionary for inference with the modified DataFrame
                self.sweep_df[stage] = df_inference

    def get_performance_pareto_one_stage(
        self,
        stage,
        num_chips: int | None = None,
        num_retrieval_servers: int | None = None,
    ):
        """
        Return performance pareto of one stage.
        This function handles different stages like 'retrieval', 'rerank', 'prefill', 'decode', etc.
        It also calculates and returns the Pareto performance for different numbers of chips or retrieval servers.
        """
        # For 'retrieval' stage, ensure num_retrieval_servers is provided
        if stage == "retrieval":
            assert num_retrieval_servers is not None
            if num_retrieval_servers not in self.performance_pareto_dict[stage]:
                performance_pareto = self.get_pareto_distributed_retrieval(
                    num_retrieval_servers=num_retrieval_servers,
                )
                self.performance_pareto_dict[stage][
                    num_retrieval_servers
                ] = performance_pareto

            return copy.deepcopy(
                self.performance_pareto_dict[stage][num_retrieval_servers]
            )

        # For 'rerank' stage, handle batch size shifting
        elif stage == "rerank":
            assert num_chips is not None
            if num_chips not in self.performance_pareto_dict[stage]:
                performance_pareto = get_filtered_df(
                    self.sweep_df[stage], {"num_chips": num_chips}
                )
                max_batch_size = np.max(performance_pareto["batch_size"].tolist())
                if max_batch_size >= self.rerank_topk:
                    performance_pareto = performance_pareto.loc[
                        performance_pareto["batch_size"] >= self.rerank_topk
                    ]
                    performance_pareto["batch_size"] = performance_pareto[
                        "batch_size"
                    ].apply(lambda x: int(x / self.rerank_topk))
                else:
                    performance_pareto = performance_pareto.loc[
                        performance_pareto["batch_size"] == max_batch_size
                    ]
                    performance_pareto["latency_s"] = performance_pareto[
                        "latency_s"
                    ].apply(lambda x: int(x * self.rerank_topk / max_batch_size))
                    performance_pareto["qps"] = performance_pareto["qps"].apply(
                        lambda x: int(x * max_batch_size / self.rerank_topk)
                    )
                    performance_pareto["qps_per_chip"] = performance_pareto[
                        "qps_per_chip"
                    ].apply(lambda x: int(x * max_batch_size / self.rerank_topk))
                    performance_pareto["batch_size"] = 1

                self.performance_pareto_dict[stage][num_chips] = performance_pareto

            return copy.deepcopy(self.performance_pareto_dict[stage][num_chips])

        # For other stages like 'prefill', 'decode', etc.
        else:
            assert num_chips is not None
            if num_chips not in self.performance_pareto_dict[stage]:
                performance_pareto = get_filtered_df(
                    self.sweep_df[stage], {"num_chips": num_chips}
                )
                self.performance_pareto_dict[stage][num_chips] = performance_pareto

            return copy.deepcopy(self.performance_pareto_dict[stage][num_chips])

    def get_possible_batch_sizes_one_stage(
        self,
        stage,
        max_batch_size_request: int | None = None,
        num_chips: int | None = None,
        num_retrieval_servers: int | None = None,
    ):
        """
        Return possible batch sizes of Pareto for one stage. Note that the sizes
        may not reach the max_batch_size_request.
        """
        if stage == "encode":
            performance_pareto_one_chip = self.get_performance_pareto_one_stage(
                stage,
                num_chips=1,
                num_retrieval_servers=num_retrieval_servers,
            )
            max_encoder_batch_size_one_chip = performance_pareto_one_chip[
                "batch_size"
            ].max()
            workload_per_query_one_chip = self.encode_db_nvec / num_chips
            if workload_per_query_one_chip > max_encoder_batch_size_one_chip:
                possible_batch_sizes_encode = [1]
            else:
                max_possible_batch_size_encode = int(
                    np.ceil(
                        max_encoder_batch_size_one_chip / workload_per_query_one_chip
                    )
                )
                if not is_power_of_two(max_possible_batch_size_encode):
                    max_possible_batch_size_encode *= 2
                possible_batch_sizes_encode = get_power_of_two_list(
                    max_possible_batch_size_encode
                )
            return possible_batch_sizes_encode

        else:
            # For 'retrieval', handle differently than other stages
            if stage == "retrieval":
                assert num_retrieval_servers is not None
                num_chips_or_retrieval_servers = num_retrieval_servers
            else:
                assert num_chips is not None
                num_chips_or_retrieval_servers = num_chips

            if (
                num_chips_or_retrieval_servers
                not in self.possible_batch_sizes_dict[stage]
            ):
                performance_pareto = self.get_performance_pareto_one_stage(
                    stage,
                    num_chips=num_chips,
                    num_retrieval_servers=num_retrieval_servers,
                )
                possible_batch_sizes = performance_pareto["batch_size"].tolist()
                possible_batch_sizes = get_power_of_two_list(
                    np.max(possible_batch_sizes)
                )
                self.possible_batch_sizes_dict[stage][
                    num_chips_or_retrieval_servers
                ] = possible_batch_sizes

            possible_batch_sizes = copy.deepcopy(
                self.possible_batch_sizes_dict[stage][num_chips_or_retrieval_servers]
            )
            if max_batch_size_request is not None:
                possible_batch_sizes = get_power_of_two_list(
                    np.min([max_batch_size_request, np.max(possible_batch_sizes)])
                )

            return possible_batch_sizes

    def get_latency_lookup_table_one_stage(
        self,
        stage,
        max_batch_size_request: int | None = None,
        num_chips: int | None = None,
        num_retrieval_servers: int | None = None,
    ):
        """
        Return latency lookup table of one stage.

        If max_batch_size_request is not None, the latency lookup table is ensured
        to each the maximum possible batch size.
        """
        if stage == "retrieval":
            assert num_retrieval_servers is not None
            num_chips_or_retrieval_servers = num_retrieval_servers
        else:
            assert num_chips is not None
            num_chips_or_retrieval_servers = num_chips

        needs_update = False
        if num_chips_or_retrieval_servers not in self.latency_lookup_table_dict[stage]:
            needs_update = True
        else:
            if (
                max_batch_size_request is not None
                and max_batch_size_request
                not in self.latency_lookup_table_dict[stage][
                    num_chips_or_retrieval_servers
                ]
            ):
                needs_update = True
        if needs_update:
            if max_batch_size_request is not None:
                possible_batch_sizes = get_power_of_two_list(max_batch_size_request)
            else:
                possible_batch_sizes = self.get_possible_batch_sizes_one_stage(
                    stage,
                    max_batch_size_request=max_batch_size_request,
                    num_chips=num_chips,
                    num_retrieval_servers=num_retrieval_servers,
                )

            # Get latency for encoder or other stages
            if stage == "encode":
                latency_lookup_table = {}
                for batch_size in possible_batch_sizes:
                    latency_s_encode = self.get_encode_latency_s(
                        num_chips_encode=num_chips, context_batch_size=batch_size
                    )
                    latency_lookup_table[batch_size] = latency_s_encode
            else:
                performance_pareto = self.get_performance_pareto_one_stage(
                    stage,
                    num_chips=num_chips,
                    num_retrieval_servers=num_retrieval_servers,
                )
                latency_lookup_table = {}
                for batch_size in possible_batch_sizes:
                    latency_s, _ = self.get_latency_from_pareto(
                        performance_pareto, batch_size
                    )
                    latency_lookup_table[batch_size] = latency_s

            self.latency_lookup_table_dict[stage][
                num_chips_or_retrieval_servers
            ] = latency_lookup_table

        latency_lookup_table = self.latency_lookup_table_dict[stage][
            num_chips_or_retrieval_servers
        ]

        return copy.deepcopy(latency_lookup_table)

    def get_latency_from_pareto(
        self, df: pd.DataFrame, batch_size: int, constraints: dict[str, Any] = {}
    ) -> (float, pd.DataFrame):
        """Given a Pareto dataframe, get latency from a batch size."""
        assert is_power_of_two(batch_size)
        df = get_filtered_df(df, constraints)
        possible_batch_sizes = list(set(df["batch_size"].tolist()))
        if batch_size in possible_batch_sizes:
            selected_row = df.loc[df["batch_size"] == batch_size]
            latency_s = selected_row["latency_s"].tolist()
        else:
            if batch_size < np.min(possible_batch_sizes):
                new_batch_size = np.min(possible_batch_sizes)
                assert is_power_of_two(new_batch_size)
                selected_row = df.loc[df["batch_size"] == new_batch_size]
                latency_s = selected_row["latency_s"].tolist()
            else:
                new_batch_size = np.max(possible_batch_sizes)
                assert is_power_of_two(new_batch_size)
                selected_row = df.loc[df["batch_size"] == new_batch_size]
                latency_s = selected_row["latency_s"].tolist()
                latency_s = [
                    latency * batch_size / new_batch_size for latency in latency_s
                ]
        assert len(latency_s) == 1
        latency_s = latency_s[0]

        return latency_s, selected_row

    def get_pareto_distributed_retrieval(
        self,
        num_retrieval_servers: int = 1,
    ) -> pd.DataFrame:
        """
        Given the number of retrieval servers, find the per-Pareto retrieval.

        Basically, it applies the latency based on the number of retrieval servers
            closest to the given server number.
        """
        assert num_retrieval_servers >= 1
        perf_df = copy.deepcopy(self.sweep_df["retrieval"])
        num_retrieval_servers_in_df = sorted(perf_df["num_servers"].unique())
        if num_retrieval_servers < min(num_retrieval_servers_in_df):
            raise ValueError(
                f"Does not support retrieval less than {min(num_retrieval_servers_in_df)} servers"
            )

        closest_num_retrieval_servers = num_retrieval_servers_in_df[0]
        for i, n in enumerate(num_retrieval_servers_in_df):
            if n <= num_retrieval_servers:
                # Get the closest number of retrieval servers <= specified one
                if (
                    i == len(num_retrieval_servers_in_df) - 1
                    or num_retrieval_servers_in_df[i + 1] > num_retrieval_servers
                ):
                    closest_num_retrieval_servers = n

        # Apply the latency calculation
        perf_df = get_filtered_df(
            perf_df, {"num_servers": closest_num_retrieval_servers}
        )

        # Get Pareto frontier based on latency and QPS per chip
        retrieval_performance_pareto = get_pareto_df(
            sweep_df=perf_df,
            keys_lower_is_better=[("latency_s", True), ("qps_per_chip", False)],
        )
        return retrieval_performance_pareto

    def get_encode_latency_s(
        self,
        num_chips_encode: int = 4,
        context_batch_size: int = 1,  # Number of long-context prompts to process
        constraints: dict[str, Any] = {},  # Constraints will overwrite previous params
    ):
        """
        Return the time consumption to encode db_nvec chunks.

        Here, we assume that the encoding is done in parallel on multiple chips, with
            each chip holding a copy of the encoder model (as this is a small model).
        """
        # Get performance of multiple chips
        performance_pareto_encode = get_filtered_df(
            self.sweep_df["encode"],
            {
                "num_chips": 1,
                "seq_len_inference_prefill": self.encode_seq_len_inference_prefill,
            },
        )
        performance_pareto_encode["batch_size"] = (
            performance_pareto_encode["batch_size"] * num_chips_encode
        )
        performance_pareto_encode["qps"] = (
            performance_pareto_encode["qps"] * num_chips_encode
        )
        performance_pareto_encode["num_chips"] = num_chips_encode

        # Get the largest batch size that can be used for this database
        batch_size_list = set(performance_pareto_encode["batch_size"])
        batch_size_max = min(batch_size_list)
        total_db_nvec = self.encode_db_nvec * context_batch_size
        for batch_size in batch_size_list:
            if batch_size > batch_size_max and batch_size < total_db_nvec:
                batch_size_max = batch_size

        # Choose the best-performing points, given all the filtering
        df_selected = get_filtered_df(
            performance_pareto_encode,
            {
                "batch_size": batch_size_max,
                "num_chips": num_chips_encode,
                **constraints,
            },
        )
        min_latency_per_batch = df_selected["latency_s"].min()
        latency_s_encode = min_latency_per_batch * (total_db_nvec / batch_size_max)

        return latency_s_encode

    def get_pipeline_latency_table(
        self,
        previous_stage_finish_time_LUT: dict[int, float] | None,
        batch_size_request: int,
        batch_size_current_stage: int,
        latency_per_batch_current_stage: float,
    ) -> dict[int, float]:
        """
        For the micro-batching scenario, given the finish time of a previous
        scenario, return the finish time of the current stage (req_id: end_time).
        """
        if previous_stage_finish_time_LUT is None:
            previous_stage_finish_time_LUT = {}
            for qid in range(batch_size_request):
                previous_stage_finish_time_LUT[qid] = 0
        current_stage_finish_time_LUT = {}

        assert batch_size_request % batch_size_current_stage == 0
        n_batches = int(batch_size_request / batch_size_current_stage)

        for bid in range(n_batches):
            start_qid = bid * batch_size_current_stage
            end_qid = start_qid + batch_size_current_stage

            # Get the start time of the current batch
            if start_qid > 0:
                start_time_this_stage = current_stage_finish_time_LUT[start_qid - 1]
            else:
                start_time_this_stage = 0
            finish_time_last_stage = previous_stage_finish_time_LUT[end_qid - 1]
            start_time = np.max([start_time_this_stage, finish_time_last_stage])

            end_time = start_time + latency_per_batch_current_stage
            for qid in range(start_qid, end_qid):
                current_stage_finish_time_LUT[qid] = end_time

        return current_stage_finish_time_LUT

    def generate_collocated_execution_schedule(
        self, batch_sizes, max_batch_size_request=None
    ):
        """
        Generate an execution schedule for a given set of stages with varying batch sizes.

        Parameters:
            batch_sizes (list of int): List of batch sizes corresponding to each stage, e.g., [4, 2, 8].

        Returns:
            list: A schedule indicating the execution sequence of the stage ids.
        """

        def recursive_partial_schedule(batch_sizes, start_stage_id):
            """
            Recursive function that generates a partial schedule until batch sizes are not monolithically smaller.

            Parameters:
                batch_sizes (list of int): List of batch sizes for each stage.
                start_stage_id (int): The current stage ID to start the schedule from.

            Returns:
                tuple: (partial schedule until batch sizes no longer strictly decrease, end stage ID)
            """
            partial_schedule = [start_stage_id]
            current_stage_id = start_stage_id
            next_stage_id = start_stage_id + 1
            until_stage_id = current_stage_id

            # Iterate through stages until the batch size is no longer smaller
            if next_stage_id < len(batch_sizes):
                current_batch_size = batch_sizes[current_stage_id]
                next_batch_size = batch_sizes[next_stage_id]

                if next_batch_size <= current_batch_size:
                    upcoming_schedule, until_stage_id = recursive_partial_schedule(
                        batch_sizes, next_stage_id
                    )
                    num_repeats = current_batch_size // next_batch_size
                    partial_schedule += num_repeats * list(upcoming_schedule)
                else:
                    pass

            return partial_schedule, until_stage_id

        # Initialize the execution schedule and counters for each stage
        schedule = []
        current_stage_id = 0
        while current_stage_id < len(batch_sizes) - 1:
            next_stage_id = current_stage_id + 1
            if current_stage_id == 0:
                schedule = [current_stage_id]
            current_batch_size = np.max(batch_sizes[: current_stage_id + 1])
            next_batch_size = batch_sizes[next_stage_id]

            # Determine how many times the current stage needs to run to align with the next stage
            if next_batch_size > current_batch_size:
                num_repeats = next_batch_size // current_batch_size
                schedule = num_repeats * schedule
                schedule.append(next_stage_id)
                current_stage_id = next_stage_id
            else:
                num_repeats = current_batch_size // next_batch_size
                partial_schedule, reached_stage_id = recursive_partial_schedule(
                    batch_sizes, next_stage_id
                )
                schedule += num_repeats * partial_schedule
                current_stage_id = reached_stage_id

        max_batch_size = np.max(batch_sizes)
        total_batch_size_per_stage = np.zeros(len(batch_sizes))
        for stage_id in schedule:
            total_batch_size_per_stage[stage_id] += batch_sizes[stage_id]
        for stage_id in range(len(batch_sizes)):
            assert total_batch_size_per_stage[stage_id] == max_batch_size

        if max_batch_size_request is not None:
            num_repeats = max_batch_size_request // np.max(batch_sizes)
            schedule = num_repeats * schedule

        return schedule

    def get_collocated_latency_table(
        self,
        previous_stage_finish_time_LUT: dict[int, float] | None,
        batch_size_request: int,
        batch_sizes_current_stages: list[int],
        latency_per_batch_current_stages: list[float],
    ) -> dict[int, float]:
        """
        For the micro-batching scenario, given the finish time of a previous
        scenario, return the finish time of the current stage (req_id: end_time).
        """
        assert len(batch_sizes_current_stages) == len(latency_per_batch_current_stages)
        num_stages = len(batch_sizes_current_stages)

        if num_stages == 1:
            return self.get_pipeline_latency_table(
                previous_stage_finish_time_LUT=previous_stage_finish_time_LUT,
                batch_size_request=batch_size_request,
                batch_size_current_stage=batch_sizes_current_stages[0],
                latency_per_batch_current_stage=latency_per_batch_current_stages[0],
            )

        if previous_stage_finish_time_LUT is None:
            previous_stage_finish_time_LUT = {}
            for qid in range(batch_size_request):
                previous_stage_finish_time_LUT[qid] = 0
        current_stage_finish_time_LUT = {}

        schedule = self.generate_collocated_execution_schedule(
            batch_sizes_current_stages, max_batch_size_request=batch_size_request
        )
        start_id_first_stage = 0
        end_id_last_stage = 0
        current_timeline = 0

        for stage_id in schedule:
            if stage_id == 0:
                finish_id = start_id_first_stage + batch_sizes_current_stages[stage_id]
                start_time = np.max(
                    [current_timeline, previous_stage_finish_time_LUT[finish_id - 1]]
                )
                start_id_first_stage += batch_sizes_current_stages[stage_id]
            else:
                start_time = current_timeline
            current_timeline = start_time + latency_per_batch_current_stages[stage_id]
            if stage_id == num_stages - 1:
                for qid in range(
                    end_id_last_stage,
                    end_id_last_stage + batch_sizes_current_stages[stage_id],
                ):
                    current_stage_finish_time_LUT[qid] = current_timeline
                end_id_last_stage += batch_sizes_current_stages[stage_id]

        return current_stage_finish_time_LUT

    def get_collocation_combinations(
        self,
        stage_names,
        filter_full_disaggregation=True,
        filter_retrieval_at_beginning_or_end=True,
        disaggregate_decode=True,
    ):
        """
        Given a list of stage names, return a list of all possible collocated stages.
        This should be the top-level function to call.
        """
        if self.all_placement_strategies is not None:
            return self.all_placement_strategies

        def get_collocation_combinations_recursive(stage_names):
            """
            Given a list of stage names, return a list of all possible collocated stages.
            This function should only be called by get_collocation_combinations.
            """
            placement_stages = []
            for i in range(1, len(stage_names)):
                first_stage = stage_names[:i]
                if len(stage_names[i:]) > 1:
                    all_rest_stages = get_collocation_combinations_recursive(
                        stage_names[i:]
                    )
                    for rest_stages in all_rest_stages:
                        placement_stages.append([first_stage] + rest_stages)
                elif len(stage_names[i:]) == 1:
                    placement_stages.append([first_stage] + [stage_names[i:]])
                    placement_stages.append([stage_names])
                else:
                    placement_stages.append([first_stage])

            return placement_stages

        all_placement_possibilities = get_collocation_combinations_recursive(
            stage_names
        )
        all_placement_strategies = []
        for i, placement_stages in enumerate(all_placement_possibilities):
            keep_mask = True
            if filter_full_disaggregation and len(placement_stages) == len(stage_names):
                keep_mask = False
            if disaggregate_decode:
                if (
                    len(placement_stages[-1]) != 1
                    or placement_stages[-1][0] != "decode"
                ):
                    keep_mask = False
            for collocated_stages in placement_stages:
                if filter_retrieval_at_beginning_or_end and len(collocated_stages) > 1:
                    if (
                        collocated_stages[0] == "retrieval"
                        or collocated_stages[-1] == "retrieval"
                    ):
                        keep_mask = False

            if keep_mask:
                all_placement_strategies.append(placement_stages)

        self.all_placement_strategies = all_placement_strategies

        return all_placement_strategies

    def get_result_df_columns(self):
        columns = []
        columns += ["latency_s", "latency_s_ttft", "latency_s_tpot"]
        columns += ["qps_per_chip", "qps", "qps_bound"]
        columns += [f"qps_{stage}" for stage in self.stages]
        columns += ["batch_size_request"]
        columns += [f"batch_size_{stage}" for stage in self.stages]
        columns += ["num_chips_e2e"]
        columns += [
            f"num_chips_{stage}" for stage in self.stages if stage != "retrieval"
        ]
        columns += ["num_retrieval_servers"]
        columns += ["placement_policy", "collocation_strategy", "scheduling_policy"]
        columns += [
            "placement_and_resource_id"
        ]  # The placement + resource allocation combination
        return columns

    """ Helper Functions Ends """

    """ Cost assembling for a single hardware starts """

    def assemble_cost(
        self,
        physical_mapping: PhysicalMapping = PhysicalMapping(),
        scheduling_policy: str = "continuous-batching",
        max_batch_size_request: int = 1024,
        no_microbatching: bool = True,
    ):
        placement_policy = physical_mapping.placement_policy
        assert placement_policy in ["disaggregated", "collocated"]
        assert scheduling_policy in ["continuous-batching"]

        # Define the result dataframe
        columns = self.get_result_df_columns()

        for stage in self.stages:
            if stage != "retrieval":
                assert physical_mapping.num_chips[stage]
                self.scale_up_performance_to_more_chips(
                    stage,
                    max_num_chips=physical_mapping.num_chips[stage],
                    max_num_retrieval_servers=None,
                )
            else:
                assert physical_mapping.num_retrieval_servers
                self.scale_up_performance_to_more_chips(
                    stage,
                    max_num_chips=None,
                    max_num_retrieval_servers=physical_mapping.num_retrieval_servers,
                )

        # Find collocated stages
        if physical_mapping.collocation_strategy is not None:
            all_placement_strategies = [physical_mapping.collocation_strategy]
        else:
            if placement_policy == "collocated":
                all_placement_strategies = self.get_collocation_combinations(
                    self.stages
                )
            elif placement_policy == "disaggregated":
                all_placement_strategies = [[[stage] for stage in self.stages]]
        print("All valid collocation strategies: ", all_placement_strategies)

        time_start = time.time()
        new_rows = []

        for collocation_strategy in all_placement_strategies:
            # Get the number of total chips involved
            num_chips = physical_mapping.num_chips
            num_retrieval_servers = physical_mapping.num_retrieval_servers

            num_chips_e2e = 0
            for collocated_stage_names in collocation_strategy:
                num_chips_this_stage = np.max(
                    [
                        num_chips[stage] if stage != "retrieval" else 0
                        for stage in collocated_stage_names
                    ]
                )
                num_chips_e2e += num_chips_this_stage
            num_chips["e2e"] = num_chips_e2e

            possible_batch_sizes_dict = {}
            for stage in self.stages:
                possible_batch_sizes_dict[stage] = (
                    self.get_possible_batch_sizes_one_stage(
                        stage,
                        max_batch_size_request=(
                            max_batch_size_request if stage != "decode" else None
                        ),
                        num_chips=num_chips[stage] if stage != "retrieval" else None,
                        num_retrieval_servers=num_retrieval_servers,
                    )
                )

            latency_table_dict = {}
            for stage in self.stages:
                latency_table_dict[stage] = self.get_latency_lookup_table_one_stage(
                    stage,
                    max_batch_size_request=(
                        max_batch_size_request
                        if stage != "decode"
                        else np.max(possible_batch_sizes_dict[stage])
                    ),
                    num_chips=num_chips[stage] if stage != "retrieval" else None,
                    num_retrieval_servers=num_retrieval_servers,
                )
                print(
                    "stage: ", stage, "latency_table_dict: ", latency_table_dict[stage]
                )

            # Handle microbatching or non-microbatching cases
            if no_microbatching:
                possible_batch_sizes_request = get_power_of_two_list(
                    max_batch_size_request
                )
                batch_sizes_combinations_two_stages = product(
                    possible_batch_sizes_request, possible_batch_sizes_dict["decode"]
                )

                batch_sizes_combinations_before_filtering = []
                for batch_size_per_stage in batch_sizes_combinations_two_stages:
                    batch_size_request = batch_size_per_stage[0]
                    batch_size_decode = batch_size_per_stage[-1]
                    if batch_size_request <= batch_size_decode:
                        batch_sizes_collocated = [
                            batch_size_request for _ in range(len(self.stages) - 1)
                        ] + [batch_size_decode]
                        batch_sizes_combinations_before_filtering.append(
                            batch_sizes_collocated
                        )
            # With microbatching: all stages can use heterogeneous batch sizes
            else:
                params = [possible_batch_sizes_dict[stage] for stage in self.stages]
                batch_sizes_combinations_before_filtering = product(*params)

            # 3D array, each element is a 2D array of collocated batch sizes of the stages
            batch_sizes_collocated_combinations = []
            print(
                "batch_sizes_combinations_before_filtering: ",
                batch_sizes_combinations_before_filtering,
            )
            for batch_size_per_stage in batch_sizes_combinations_before_filtering:
                batch_size_request = np.max(batch_size_per_stage[:-1])
                batch_size_decode = batch_size_per_stage[-1]

                if batch_size_request <= batch_size_decode:
                    # Convert to list and initialize the list for collocated batch sizes
                    batch_size_per_stage = list(batch_size_per_stage)
                    batch_sizes_collocated = []

                    i = 0
                    # Split the batch sizes for each collocated stage in the strategy
                    for _, collocated_stages in enumerate(collocation_strategy):
                        batch_sizes_collocated.append(
                            batch_size_per_stage[i : i + len(collocated_stages)]
                        )
                        i = i + len(collocated_stages)

                    batch_sizes_collocated_combinations.append(batch_sizes_collocated)

            # Initialize performance cache
            qps_per_collocated_stages = []
            qps_per_single_stage = []
            previous_stage_finish_time_LUT = None

            qps_per_collocated_stages_dict = {"perf": qps_per_collocated_stages}
            qps_per_single_stage_dict = {"perf": qps_per_single_stage}
            previous_stage_finish_time_LUT_dict = {
                "perf": previous_stage_finish_time_LUT
            }

            new_rows = []
            # Process each batch size combination
            for (
                batch_sizes_collocated_combination
            ) in batch_sizes_collocated_combinations:
                flatten_batch_size = []
                for b in batch_sizes_collocated_combination:
                    flatten_batch_size.extend(b)

                batch_size_request = np.max(flatten_batch_size[:-1])

                # Refer to the original cache dictionary
                qps_per_collocated_stages_dict_tmp = qps_per_collocated_stages_dict
                qps_per_single_stage_dict_tmp = qps_per_single_stage_dict
                previous_stage_finish_time_LUT_dict_tmp = (
                    previous_stage_finish_time_LUT_dict
                )

                # Process each collocated stage
                for batch_size_per_collocated_stage, collocated_stage_names in zip(
                    batch_sizes_collocated_combination, collocation_strategy
                ):
                    cache_key = tuple(batch_size_per_collocated_stage)

                    # Check cache before recalculating
                    if (
                        cache_key in qps_per_collocated_stages_dict_tmp
                        and cache_key in qps_per_single_stage_dict_tmp
                        and cache_key in previous_stage_finish_time_LUT_dict_tmp
                    ):
                        qps_per_collocated_stages_dict_tmp = (
                            qps_per_collocated_stages_dict_tmp[cache_key]
                        )
                        qps_per_single_stage_dict_tmp = qps_per_single_stage_dict_tmp[
                            cache_key
                        ]
                        previous_stage_finish_time_LUT_dict_tmp = (
                            previous_stage_finish_time_LUT_dict_tmp[cache_key]
                        )
                        continue
                    else:
                        # Cache miss, calculate the performance for this batch size
                        qps_per_collocated_stages = copy.deepcopy(
                            qps_per_collocated_stages_dict_tmp["perf"]
                        )
                        qps_per_single_stage = copy.deepcopy(
                            qps_per_single_stage_dict_tmp["perf"]
                        )
                        previous_stage_finish_time_LUT = copy.deepcopy(
                            previous_stage_finish_time_LUT_dict_tmp["perf"]
                        )

                        # Update the cache
                        qps_per_collocated_stages_dict_tmp[cache_key] = {}
                        qps_per_single_stage_dict_tmp[cache_key] = {}
                        previous_stage_finish_time_LUT_dict_tmp[cache_key] = {}

                    # Get latency for the current collocated stages
                    latency_per_batch_current_stages = [
                        latency_table_dict[stage_name][
                            batch_size_per_collocated_stage[i]
                        ]
                        for i, stage_name in enumerate(collocated_stage_names)
                    ]

                    qps_current_stages = [
                        (
                            batch_size_per_collocated_stage[i]
                            / latency_per_batch_current_stages[i]
                            if latency_per_batch_current_stages[i] != 0
                            else np.inf
                        )
                        for i in range(len(collocated_stage_names))
                    ]

                    qps_per_single_stage += qps_current_stages
                    overall_qps_current_stages = 1 / np.sum(
                        1 / np.array(qps_current_stages)
                    )
                    qps_per_collocated_stages.append(overall_qps_current_stages)

                    # Normal collocation stages
                    if "decode" != collocated_stage_names[-1]:
                        current_stage_finish_time_LUT = self.get_collocated_latency_table(
                            previous_stage_finish_time_LUT=previous_stage_finish_time_LUT,
                            batch_size_request=max_batch_size_request,
                            batch_sizes_current_stages=batch_size_per_collocated_stage,
                            latency_per_batch_current_stages=latency_per_batch_current_stages,
                        )
                        previous_stage_finish_time_LUT = current_stage_finish_time_LUT

                        # Update the cache with performance results
                        qps_per_collocated_stages_dict_tmp[cache_key]["perf"] = (
                            copy.deepcopy(qps_per_collocated_stages)
                        )
                        qps_per_single_stage_dict_tmp[cache_key]["perf"] = (
                            copy.deepcopy(qps_per_single_stage)
                        )
                        previous_stage_finish_time_LUT_dict_tmp[cache_key]["perf"] = (
                            copy.deepcopy(previous_stage_finish_time_LUT)
                        )

                        qps_per_collocated_stages_dict_tmp = (
                            qps_per_collocated_stages_dict_tmp[cache_key]
                        )
                        qps_per_single_stage_dict_tmp = qps_per_single_stage_dict_tmp[
                            cache_key
                        ]
                        previous_stage_finish_time_LUT_dict_tmp = (
                            previous_stage_finish_time_LUT_dict_tmp[cache_key]
                        )

                    else:  # Decode is the last stage
                        assert "decode" == collocated_stage_names[-1]

                        batch_size_decode = batch_size_per_collocated_stage[-1]
                        latency_s_decode = latency_table_dict["decode"][
                            batch_size_decode
                        ]

                        # Pop the decode part and compute the other collocated stages
                        batch_size_per_collocated_stage = (
                            batch_size_per_collocated_stage[:-1]
                        )
                        latency_per_batch_current_stages = (
                            latency_per_batch_current_stages[:-1]
                        )

                        # Process stages before decode
                        if len(latency_per_batch_current_stages) >= 1:
                            current_stage_finish_time_LUT = self.get_collocated_latency_table(
                                previous_stage_finish_time_LUT=previous_stage_finish_time_LUT,
                                batch_size_request=max_batch_size_request,
                                batch_sizes_current_stages=batch_size_per_collocated_stage,
                                latency_per_batch_current_stages=latency_per_batch_current_stages,
                            )
                            previous_stage_finish_time_LUT = (
                                current_stage_finish_time_LUT
                            )

                            qps_decode = batch_size_decode / latency_s_decode
                            qps_current_stages_non_decode = qps_current_stages[:-1]
                            overall_qps_current_stages_non_decode = 1 / np.sum(
                                1 / np.array(qps_current_stages_non_decode)
                            )

                            # Calculate time percentage
                            time_perc_dec = (
                                overall_qps_current_stages
                                / overall_qps_current_stages_non_decode
                            )
                            time_perc_non_dec = overall_qps_current_stages / qps_decode

                            latency_s_decode_normalized = (
                                latency_s_decode / time_perc_dec
                            )

                            # Time between tokens and time to first token
                            latency_s_tpot = (
                                latency_s_decode_normalized / self.dec_steps
                            )
                            latency_s_ttft = np.average(
                                [
                                    previous_stage_finish_time_LUT[i]
                                    for i in range(batch_size_request)
                                ]
                            )

                            latency_s_e2e = latency_s_ttft + latency_s_decode_normalized

                        else:  # Only decode, no collocation
                            latency_s_tpot = latency_s_decode / self.dec_steps
                            latency_s_ttft = np.average(
                                [
                                    previous_stage_finish_time_LUT[i]
                                    for i in range(batch_size_request)
                                ]
                            )
                            latency_s_e2e = latency_s_ttft + latency_s_decode

                        qps = np.min(qps_per_collocated_stages)
                        qps_per_chip = qps / num_chips["e2e"]
                        qps_bound = None
                        for sid in range(len(qps_per_collocated_stages)):
                            if qps == qps_per_collocated_stages[sid]:
                                qps_bound = collocation_strategy[sid]

                        result_dict = {
                            "latency_s": latency_s_e2e,
                            "latency_s_ttft": latency_s_ttft,
                            "latency_s_tpot": latency_s_tpot,
                            "qps_per_chip": qps_per_chip,
                            "qps": qps,
                            "qps_bound": qps_bound,
                            "batch_size_request": batch_size_request,
                            "num_chips_e2e": num_chips["e2e"],
                            "num_retrieval_servers": num_retrieval_servers,
                            "collocation_strategy": str(collocation_strategy),
                            "placement_and_resource_id": 0,
                        }
                        for i, stage_name in enumerate(self.stages):
                            result_dict[f"qps_{stage_name}"] = qps_per_single_stage[i]
                            result_dict[f"batch_size_{stage_name}"] = (
                                flatten_batch_size[i]
                            )
                            if stage_name != "retrieval":
                                result_dict[f"num_chips_{stage_name}"] = num_chips[
                                    stage_name
                                ]
                        new_rows.append(result_dict)

            print(f"assemble_cost logic time: {time.time() - time_start}")
            time_start = time.time()
            if len(new_rows) == 0:
                return pd.DataFrame()
            new_row_keys = set(new_rows[0].keys())
            merged_new_rows = {key: [] for key in new_row_keys}
            for new_row in new_rows:
                assert set(new_row.keys()) == new_row_keys
                for key in new_row_keys:
                    merged_new_rows[key].append(new_row[key])
            print(f"dict merge time: {time.time() - time_start}")
            time_start = time.time()
            df_e2e_perf = pd.DataFrame(merged_new_rows, columns=columns)
            print(f"df creation time: {time.time() - time_start}")
            time_start = time.time()
            df_e2e_perf["placement_policy"] = placement_policy
            df_e2e_perf["scheduling_policy"] = scheduling_policy

            print(f"post processing time: {time.time() - time_start}")

        return df_e2e_perf

    """ Cost assembling for a single hardware ends """

    """ Sweep performance across hardware starts """

    def run_sweep_get_hardware_mappings(
        self,
        min_num_chips: int = 4,
        max_num_chips: int = 128,
        max_num_chips_per_stage: int = 64,
        placement_policy: str = "disaggregated",  # 'collocated' or 'disaggregated'
        collocation_strategy: list[list[str]] | None = None,
        possible_num_chips: dict[str, int] = {
            "rewrite_prefill": None,
            "rewrite_decode": None,
            "encode": None,
            "rerank": None,
            "prefill": None,
            "decode": None,
        },
    ) -> list[PhysicalMapping]:
        """
        Get a list of possible hardware mappings.

        Assuming all host servers for prefill and decode are used for retrieval.
        """
        # make sure enough min chips for retrieval
        min_chips_retrieval = int(
            self.sweep_df["retrieval"]["num_servers"].min() * self.num_chips_per_server
        )
        if min_num_chips < min_chips_retrieval:
            min_num_chips = min_chips_retrieval
            assert (
                min_num_chips <= max_num_chips
            ), "min_num_chips must be less than max_num_chips"

        possible_num_chips_dict = {}
        for stage in self.stages:
            if stage == "retrieval":
                self.scale_up_performance_to_more_chips(
                    stage,
                    max_num_chips=None,
                    max_num_retrieval_servers=int(
                        max_num_chips / self.num_chips_per_server
                    ),
                )
            elif possible_num_chips[stage] is not None:
                possible_num_chips_dict[stage] = sorted(possible_num_chips[stage])
                self.scale_up_performance_to_more_chips(
                    stage,
                    max_num_chips=np.max(possible_num_chips_dict[stage]),
                    max_num_retrieval_servers=None,
                )
                possible_num_chips_dict[stage] = [
                    nc
                    for nc in possible_num_chips_dict[stage]
                    if nc <= max_num_chips_per_stage
                ]
            else:
                self.scale_up_performance_to_more_chips(
                    stage,
                    max_num_chips=max_num_chips_per_stage,
                    max_num_retrieval_servers=None,
                )
                if stage == "encode" or stage == "rerank":
                    possible_num_chips_dict[stage] = get_power_of_two_list(
                        max_num_chips_per_stage
                    )
                else:
                    possible_num_chips_dict[stage] = sorted(
                        set(self.sweep_df[stage]["num_chips"].tolist())
                    )
                possible_num_chips_dict[stage] = [
                    nc
                    for nc in possible_num_chips_dict[stage]
                    if nc <= max_num_chips_per_stage
                ]

        if placement_policy == "disaggregated":
            previous_num_chips = 0
            previous_mapping_and_num_chips = [(PhysicalMapping(), previous_num_chips)]
            for stage in self.stages:
                if stage == "retrieval":
                    continue
                possible_num_chips_current_stage = possible_num_chips_dict[stage]
                current_mapping_and_num_chips = []
                for (
                    physical_mapping,
                    previous_num_chips,
                ) in previous_mapping_and_num_chips:
                    for current_num_chips in possible_num_chips_current_stage:
                        current_physical_mapping = copy.deepcopy(physical_mapping)
                        current_physical_mapping.num_chips[stage] = current_num_chips
                        num_chips = current_num_chips + previous_num_chips
                        current_mapping_and_num_chips.append(
                            (current_physical_mapping, num_chips)
                        )
                previous_mapping_and_num_chips = current_mapping_and_num_chips

            mapping_and_num_chips_before_filtering = previous_mapping_and_num_chips

        elif placement_policy == "collocated":
            previous_num_chips = 0
            previous_mapping_and_num_chips = [(PhysicalMapping(), previous_num_chips)]
            for collocated_stages in collocation_strategy:
                if len(collocated_stages) == 1 and collocated_stages[0] == "retrieval":
                    continue

                possible_num_chips_collocate = set(
                    get_power_of_two_list(max_num_chips_per_stage)
                )
                for stage in collocated_stages:
                    if stage == "retrieval":
                        continue
                    possible_num_chips_collocate = (
                        possible_num_chips_collocate.intersection(
                            possible_num_chips_dict[stage]
                        )
                    )
                possible_num_chips_collocate = sorted(possible_num_chips_collocate)

                current_mapping_and_num_chips = []
                for (
                    physical_mapping,
                    previous_num_chips,
                ) in previous_mapping_and_num_chips:
                    for current_num_chips in possible_num_chips_collocate:
                        current_physical_mapping = copy.deepcopy(physical_mapping)
                        for stage in collocated_stages:
                            if stage == "retrieval":
                                continue
                            current_physical_mapping.num_chips[stage] = (
                                current_num_chips
                            )
                        num_chips = previous_num_chips + current_num_chips
                        current_mapping_and_num_chips.append(
                            (current_physical_mapping, num_chips)
                        )
                previous_mapping_and_num_chips = current_mapping_and_num_chips

            mapping_and_num_chips_before_filtering = previous_mapping_and_num_chips

        else:
            raise ValueError(f"Unknown mapping policy: {placement_policy}")

        # Regardless of task placement, filter unqualified mappings, and add retrieval servers
        possible_physical_mappings = []
        for physical_mapping, total_num_chips in mapping_and_num_chips_before_filtering:
            if total_num_chips > max_num_chips or total_num_chips < min_num_chips:
                continue
            num_retrieval_servers = int(total_num_chips / self.num_chips_per_server)
            if num_retrieval_servers < 1:
                num_retrieval_servers = 1
            physical_mapping.num_retrieval_servers = num_retrieval_servers
            physical_mapping.num_chips["e2e"] = total_num_chips
            physical_mapping.placement_policy = placement_policy
            physical_mapping.collocation_strategy = collocation_strategy
            possible_physical_mappings.append(physical_mapping)

        return possible_physical_mappings

    def run_sweep_across_systems(
        self,
        min_num_chips: int = 64,
        max_num_chips: int = 128,
        max_num_chips_per_stage: int = 64,
        placement_policy: str = "collocated",
        scheduling_policy: str = "continuous-batching",
        max_batch_size_request: int = 1024,
        no_microbatching: bool = True,
        verbose_microbatching_options=False,  # if true, allow request batch size > min(stage batch sizes before decode)
        possible_num_chips: dict[str, int] = {
            "rewrite_prefill": None,
            "rewrite_decode": None,
            "encode": None,
            "rerank": None,
            "prefill": None,
            "decode": None,
        },
    ):
        assert placement_policy in ["disaggregated", "collocated"]
        assert scheduling_policy in ["continuous-batching"]

        # Define the result dataframe
        columns = self.get_result_df_columns()

        # Find collocated stages
        if placement_policy == "collocated":
            all_placement_strategies = self.get_collocation_combinations(self.stages)
        elif placement_policy == "disaggregated":
            all_placement_strategies = [[[stage] for stage in self.stages]]
        print("All valid collocation strategies: ", all_placement_strategies)

        time_start = time.time()
        new_rows = []
        placement_and_resource_id = 0

        for strategy in all_placement_strategies:
            hardware_mappings = self.run_sweep_get_hardware_mappings(
                min_num_chips=min_num_chips,
                max_num_chips=max_num_chips,
                max_num_chips_per_stage=max_num_chips_per_stage,
                placement_policy=placement_policy,
                collocation_strategy=strategy,
                possible_num_chips=possible_num_chips,
            )
            last_first_stage_num_chips = None
            last_first_stage_num_retrieval_servers = None
            qps_per_collocated_stages_dict = None
            qps_per_single_stage_dict = None
            previous_stage_finish_time_LUT_dict = None

            for physical_mapping in hardware_mappings:
                print("Physical mapping:", physical_mapping)
                placement_and_resource_id += 1

                # Function call of cost assembler leads to ~10x higher latency,
                # thus maintain the local code here
                # new_rows.extend(self.assemble_cost(
                #   physical_mapping=physical_mapping,
                #   collocation_strategy=strategy,
                #   scheduling_policy=scheduling_policy,
                #   max_batch_size_request=max_batch_size_request,
                #   no_microbatching=no_microbatching,
                # ))

                # Get the number of total chips involved
                num_chips = physical_mapping.num_chips
                num_retrieval_servers = physical_mapping.num_retrieval_servers

                # Cache only if the first stage is the same, otherwise can run into
                # memory consumption issue for the case4 experiment
                refresh_cache = False
                if (
                    self.stages[0] != "retrieval"
                    and num_chips[self.stages[0]] != last_first_stage_num_chips
                ):
                    refresh_cache = True
                    last_first_stage_num_chips = num_chips[self.stages[0]]
                if (
                    self.stages[0] == "retrieval"
                    and num_retrieval_servers != last_first_stage_num_retrieval_servers
                ):
                    refresh_cache = True
                    last_first_stage_num_retrieval_servers = num_retrieval_servers
                if refresh_cache:
                    # Performance cache per collocation strategy
                    qps_per_collocated_stages = []
                    qps_per_single_stage = []
                    previous_stage_finish_time_LUT = None
                    del qps_per_collocated_stages_dict
                    del qps_per_single_stage_dict
                    del previous_stage_finish_time_LUT_dict
                    gc.collect()
                    qps_per_collocated_stages_dict = {"perf": qps_per_collocated_stages}
                    qps_per_single_stage_dict = {"perf": qps_per_single_stage}
                    previous_stage_finish_time_LUT_dict = {
                        "perf": previous_stage_finish_time_LUT
                    }

                possible_batch_sizes_dict = {}
                for stage in self.stages:
                    possible_batch_sizes_dict[stage] = (
                        self.get_possible_batch_sizes_one_stage(
                            stage,
                            max_batch_size_request=(
                                max_batch_size_request if stage != "decode" else None
                            ),
                            num_chips=(
                                num_chips[stage] if stage != "retrieval" else None
                            ),
                            num_retrieval_servers=num_retrieval_servers,
                        )
                    )

                # The request size should not be larger than the max batch size per stage,
                # because larger batch sizes are used to improve throughput, but we cannot
                # improve if per-stage performance is already saturated
                # max_batch_size_request = np.min(
                #     [np.max([np.max(possible_batch_sizes_dict[stage]) for stage in self.stages if stage != 'decode']),
                #     max_batch_size_request])

                latency_table_dict = {}
                for stage in self.stages:
                    latency_table_dict[stage] = self.get_latency_lookup_table_one_stage(
                        stage,
                        max_batch_size_request=(
                            max_batch_size_request
                            if stage != "decode"
                            else np.max(possible_batch_sizes_dict[stage])
                        ),
                        num_chips=num_chips[stage] if stage != "retrieval" else None,
                        num_retrieval_servers=num_retrieval_servers,
                    )

                # No microbatching: all stages before decode use the same batch size
                if no_microbatching:
                    possible_batch_sizes_request = get_power_of_two_list(
                        max_batch_size_request
                    )
                    batch_sizes_combinations_two_stages = product(
                        possible_batch_sizes_request,
                        possible_batch_sizes_dict["decode"],
                    )

                    batch_sizes_combinations_before_filtering = []
                    for batch_size_per_stage in batch_sizes_combinations_two_stages:
                        batch_size_request = batch_size_per_stage[0]
                        batch_size_decode = batch_size_per_stage[-1]
                        if batch_size_request <= batch_size_decode:
                            batch_sizes_collocated = [
                                batch_size_request for _ in range(len(self.stages) - 1)
                            ] + [batch_size_decode]
                            batch_sizes_combinations_before_filtering.append(
                                batch_sizes_collocated
                            )
                else:
                    params = [possible_batch_sizes_dict[stage] for stage in self.stages]
                    batch_sizes_combinations_before_filtering = product(*params)

                # 3D array, each element is a 2D array that consists of collocated
                # batch sizes of the stages
                batch_sizes_collocated_combinations = []
                for batch_size_per_stage in batch_sizes_combinations_before_filtering:
                    batch_size_request = np.max(batch_size_per_stage[:-1])
                    batch_size_decode = batch_size_per_stage[-1]
                    if batch_size_request <= batch_size_decode:
                        batch_size_per_stage = list(batch_size_per_stage)
                        batch_sizes_collocated = []
                        i = 0
                        for _, collocated_stages in enumerate(strategy):
                            batch_sizes_collocated.append(
                                batch_size_per_stage[i : i + len(collocated_stages)]
                            )
                            i = i + len(collocated_stages)
                        batch_sizes_collocated_combinations.append(
                            batch_sizes_collocated
                        )

                # Assemble the cost for each batching strategy
                for (
                    batch_sizes_collocated_combination
                ) in batch_sizes_collocated_combinations:
                    flatten_batch_size = []
                    for b in batch_sizes_collocated_combination:
                        flatten_batch_size.extend(b)
                    batch_size_request = np.max(flatten_batch_size[:-1])

                    # Reference to the original dict, with proper tree depth
                    qps_per_collocated_stages_dict_tmp = qps_per_collocated_stages_dict
                    qps_per_single_stage_dict_tmp = qps_per_single_stage_dict
                    previous_stage_finish_time_LUT_dict_tmp = (
                        previous_stage_finish_time_LUT_dict
                    )

                    for batch_size_per_collocated_stage, collocated_stage_names in zip(
                        batch_sizes_collocated_combination, strategy
                    ):
                        cache_key_chips = list(
                            [
                                (
                                    num_chips[stage]
                                    if stage != "retrieval"
                                    else num_retrieval_servers
                                )
                                for stage in collocated_stage_names
                            ]
                        )
                        cache_key_batch_sizes = list(batch_size_per_collocated_stage)
                        cache_key = tuple(cache_key_chips + cache_key_batch_sizes)

                        # If performance exists, skip the current iteration
                        if (
                            cache_key in qps_per_collocated_stages_dict_tmp
                            and cache_key in qps_per_single_stage_dict_tmp
                            and cache_key in previous_stage_finish_time_LUT_dict_tmp
                        ):
                            qps_per_collocated_stages_dict_tmp = (
                                qps_per_collocated_stages_dict_tmp[cache_key]
                            )
                            qps_per_single_stage_dict_tmp = (
                                qps_per_single_stage_dict_tmp[cache_key]
                            )
                            previous_stage_finish_time_LUT_dict_tmp = (
                                previous_stage_finish_time_LUT_dict_tmp[cache_key]
                            )
                            continue

                        # Otherwise, get the performance of last stage
                        else:
                            qps_per_collocated_stages = copy.deepcopy(
                                qps_per_collocated_stages_dict_tmp["perf"]
                            )
                            qps_per_single_stage = copy.deepcopy(
                                qps_per_single_stage_dict_tmp["perf"]
                            )
                            previous_stage_finish_time_LUT = copy.deepcopy(
                                previous_stage_finish_time_LUT_dict_tmp["perf"]
                            )
                            qps_per_collocated_stages_dict_tmp[cache_key] = {}
                            qps_per_single_stage_dict_tmp[cache_key] = {}
                            previous_stage_finish_time_LUT_dict_tmp[cache_key] = {}

                        # Collect data for each collocated stage and compute performance metrics
                        latency_per_batch_current_stages = [
                            latency_table_dict[stage_name][
                                batch_size_per_collocated_stage[i]
                            ]
                            for i, stage_name in enumerate(collocated_stage_names)
                        ]
                        qps_current_stages = [
                            (
                                batch_size_per_collocated_stage[i]
                                / latency_per_batch_current_stages[i]
                                if latency_per_batch_current_stages[i] != 0
                                else np.inf
                            )
                            for i in range(len(collocated_stage_names))
                        ]
                        qps_per_single_stage += qps_current_stages
                        overall_qps_current_stages = 1 / np.sum(
                            1 / np.array(qps_current_stages)
                        )
                        qps_per_collocated_stages.append(overall_qps_current_stages)

                        # Normal collocation stages
                        if "decode" != collocated_stage_names[-1]:
                            # whether it is a disaggregated query rewriter decoder service
                            if (
                                len(collocated_stage_names) == 1
                                and collocated_stage_names[0] == "rewrite_decode"
                            ):
                                # The latency is increased by decode latency
                                current_stage_finish_time_LUT = {}
                                latency_s_rewrite_decode = (
                                    latency_per_batch_current_stages[0]
                                )
                                for (
                                    key,
                                    value,
                                ) in previous_stage_finish_time_LUT.items():
                                    current_stage_finish_time_LUT[key] = (
                                        value + latency_s_rewrite_decode
                                    )
                            else:
                                current_stage_finish_time_LUT = self.get_collocated_latency_table(
                                    previous_stage_finish_time_LUT=previous_stage_finish_time_LUT,
                                    batch_size_request=max_batch_size_request,
                                    batch_sizes_current_stages=batch_size_per_collocated_stage,
                                    latency_per_batch_current_stages=latency_per_batch_current_stages,
                                )
                            previous_stage_finish_time_LUT = (
                                current_stage_finish_time_LUT
                            )
                            # print("Adding qps_per_collocated_stages: ", qps_per_collocated_stages,
                            #       "qps_per_single_stage: ", qps_per_single_stage,
                            #       "previous_stage_finish_time_LUT: ", previous_stage_finish_time_LUT)
                            qps_per_collocated_stages_dict_tmp[cache_key]["perf"] = (
                                copy.deepcopy(qps_per_collocated_stages)
                            )
                            qps_per_single_stage_dict_tmp[cache_key]["perf"] = (
                                copy.deepcopy(qps_per_single_stage)
                            )
                            previous_stage_finish_time_LUT_dict_tmp[cache_key][
                                "perf"
                            ] = copy.deepcopy(previous_stage_finish_time_LUT)
                            qps_per_collocated_stages_dict_tmp = (
                                qps_per_collocated_stages_dict_tmp[cache_key]
                            )
                            qps_per_single_stage_dict_tmp = (
                                qps_per_single_stage_dict_tmp[cache_key]
                            )
                            previous_stage_finish_time_LUT_dict_tmp = (
                                previous_stage_finish_time_LUT_dict_tmp[cache_key]
                            )

                        else:  # decode is the last stage
                            assert (
                                "decode" == collocated_stage_names[-1]
                                and len(collocated_stage_names) == 1
                            )

                            qps = np.min(qps_per_collocated_stages)
                            qps_per_chip = qps / num_chips["e2e"]
                            qps_bound = None
                            for sid in range(len(qps_per_collocated_stages)):
                                if qps == qps_per_collocated_stages[sid]:
                                    qps_bound = strategy[sid]

                            batch_size_decode = batch_size_per_collocated_stage[-1]
                            latency_s_decode = latency_table_dict["decode"][
                                batch_size_decode
                            ]

                            # Whether to record larger possible request batch sizes
                            if verbose_microbatching_options:
                                stop_max_batch_size_request = max_batch_size_request
                            else:
                                stop_max_batch_size_request = batch_size_request

                            while True:
                                # Calculate final results
                                # time between tokens
                                latency_s_tpot = latency_s_decode / self.dec_steps
                                # time to first token (on average)
                                latency_s_ttft = np.average(
                                    [
                                        previous_stage_finish_time_LUT[i]
                                        for i in range(batch_size_request)
                                    ]
                                )
                                # end-to-end latency (on average) and throughput
                                latency_s_e2e = latency_s_ttft + latency_s_decode

                                result_dict = {
                                    "latency_s": latency_s_e2e,
                                    "latency_s_ttft": latency_s_ttft,
                                    "latency_s_tpot": latency_s_tpot,
                                    "qps_per_chip": qps_per_chip,
                                    "qps": qps,
                                    "qps_bound": qps_bound,
                                    "batch_size_request": batch_size_request,
                                    "num_chips_e2e": num_chips["e2e"],
                                    "num_retrieval_servers": num_retrieval_servers,
                                    "collocation_strategy": str(strategy),
                                    "placement_and_resource_id": placement_and_resource_id,
                                }
                                # print("written flatten_batch_size: ", flatten_batch_size, "\tbatch_size_request", batch_size_request)
                                for i, stage_name in enumerate(self.stages):
                                    result_dict[f"qps_{stage_name}"] = (
                                        qps_per_single_stage[i]
                                    )
                                    result_dict[f"batch_size_{stage_name}"] = (
                                        flatten_batch_size[i]
                                    )
                                    if stage_name != "retrieval":
                                        result_dict[f"num_chips_{stage_name}"] = (
                                            num_chips[stage_name]
                                        )
                                new_rows.append(result_dict)

                                batch_size_request *= 2
                                if batch_size_request > stop_max_batch_size_request:
                                    break

        print(f"logic time: {time.time() - time_start}")
        if len(new_rows) == 0:
            return pd.DataFrame()
        new_row_keys = set(new_rows[0].keys())
        merged_new_rows = {key: [] for key in new_row_keys}
        for new_row in new_rows:
            assert set(new_row.keys()) == new_row_keys
            for key in new_row_keys:
                merged_new_rows[key].append(new_row[key])

        df_e2e_perf = pd.DataFrame(merged_new_rows, columns=columns)
        df_e2e_perf["placement_policy"] = placement_policy
        df_e2e_perf["scheduling_policy"] = scheduling_policy

        return df_e2e_perf

    """ Sweep performance across hardware ends """
