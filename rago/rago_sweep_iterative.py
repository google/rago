"""
RAG performance analysis for iterative retrievals.
"""

import copy
import heapq
import numpy as np
import pandas as pd
import time

from typing import Any, Dict

from config import PhysicalMapping
from utils import get_power_of_two_list
from rago_sweep import RAGSweep


class IterativeRAGSweep(RAGSweep):
    """
    Class to simulate iterative retrieval in a RAG system, inheriting from RAGSweep.
    """

    def __init__(
        self,
        retrieval_policy,
        encode_db_policy,
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
        assert stages == [
            "retrieval",
            "prefill",
            "decode",
        ], "Stages must be ['retrieval', 'prefill', 'decode'], does not accept more stages"
        super().__init__(
            retrieval_policy,
            encode_db_policy,
            stages,
            sweep_df,
            seq_len_inference_prefill,
            dec_steps,
            num_chips_per_server,
        )

    def simulate_iterative_retrieval_one_run(
        self,
        batch_size_dec: int = 16,
        batch_size_initial: int = 16,
        batch_size_iterative: int = 1,
        latency_s_dec_per_token: int = 10,
        latency_s_retrieval_initial: int = 10,
        latency_s_prefill_initial: int = 10,
        latency_s_retrieval_iterative: int = 1,
        latency_s_prefill_iterative: int = 1,
        dec_steps: int = 256,
        retrieval_interval: int = 64,
        num_seq_to_simulate: int = 10,
    ):
        """
        Simulate the performance of iterative retrievals in a three-stage system:
        retrieval -> prefill -> decode
        """
        num_iter_retrieval_per_seq = int(np.floor((dec_steps - 1) / retrieval_interval))
        assert num_iter_retrieval_per_seq > 0
        latency_s_dec = latency_s_dec_per_token * dec_steps

        num_retrievals_iter = batch_size_dec * num_iter_retrieval_per_seq
        num_initial_requests = batch_size_dec / batch_size_initial
        num_iter_retrievals_between_initial = num_retrievals_iter / num_initial_requests

        dec_seq_request_queue = []
        request_tick_list = []
        for iter_id in range(num_seq_to_simulate):
            request_tick_list.append(
                np.sort(
                    np.random.uniform(
                        iter_id * latency_s_dec,
                        (iter_id + 1) * latency_s_dec,
                        (batch_size_dec, num_iter_retrieval_per_seq),
                    ),
                    axis=1,
                )
            )

        dec_seq_retrieval_time_array = np.concatenate(request_tick_list, axis=1)
        assert dec_seq_retrieval_time_array.shape == (
            batch_size_dec,
            num_seq_to_simulate * num_iter_retrieval_per_seq,
        )

        for dec_seq_id in range(batch_size_dec):
            iter_id = 0
            heapq.heappush(
                dec_seq_request_queue,
                (dec_seq_retrieval_time_array[dec_seq_id][0], dec_seq_id, iter_id),
            )

        retrieval_server_time_tick = 0
        prefill_server_time_tick = 0

        result_e2e_dec_latencys = np.zeros(batch_size_dec)
        iter_request_latencys = np.zeros(
            (batch_size_dec, num_iter_retrieval_per_seq * num_seq_to_simulate)
        )
        iter_request_start_times = np.zeros(
            (batch_size_dec, num_iter_retrieval_per_seq * num_seq_to_simulate)
        )
        iter_request_finish_times = np.zeros(
            (batch_size_dec, num_iter_retrieval_per_seq * num_seq_to_simulate)
        )

        num_processed_iter_retrieval = 0
        while dec_seq_request_queue:
            batch_iter_requests = []
            batch_iter_request_start_time = []

            while (
                len(batch_iter_requests) < batch_size_iterative
                and dec_seq_request_queue
            ):
                iter_request_start_time, dec_seq_id, iter_id = heapq.heappop(
                    dec_seq_request_queue
                )
                batch_iter_requests.append(
                    (iter_request_start_time, dec_seq_id, iter_id)
                )
                batch_iter_request_start_time.append(iter_request_start_time)
                iter_request_start_times[dec_seq_id][iter_id] = iter_request_start_time

            num_processed_iter_retrieval += len(batch_iter_requests)
            while num_processed_iter_retrieval >= num_iter_retrievals_between_initial:
                retrieval_server_time_tick += latency_s_retrieval_initial
                prefill_server_time_tick += latency_s_prefill_initial
                num_processed_iter_retrieval -= num_iter_retrievals_between_initial

            retrieval_start_time = np.max(
                batch_iter_request_start_time + [retrieval_server_time_tick]
            )
            retrieval_finish_time = retrieval_start_time + latency_s_retrieval_iterative
            retrieval_server_time_tick = retrieval_finish_time

            prefill_start_time = np.max([retrieval_finish_time, prefill_server_time_tick])
            prefill_finish_time = prefill_start_time + latency_s_prefill_iterative
            prefill_server_time_tick = prefill_finish_time

            current_time = prefill_finish_time

            for iter_request_start_time, dec_seq_id, iter_id in batch_iter_requests:
                if iter_id + 1 < num_iter_retrieval_per_seq * num_seq_to_simulate:
                    time_to_next_request = (
                        dec_seq_retrieval_time_array[dec_seq_id][iter_id + 1]
                        - dec_seq_retrieval_time_array[dec_seq_id][iter_id]
                    )
                    next_iter_request_start_time = current_time + time_to_next_request
                    heapq.heappush(
                        dec_seq_request_queue,
                        (next_iter_request_start_time, dec_seq_id, iter_id + 1),
                    )
                else:
                    time_to_finish = (
                        num_seq_to_simulate * latency_s_dec
                        - dec_seq_retrieval_time_array[dec_seq_id][iter_id]
                    )
                    result_e2e_dec_latencys[dec_seq_id] = current_time + time_to_finish

                request_latency = current_time - iter_request_start_time
                iter_request_finish_times[dec_seq_id][iter_id] = current_time
                iter_request_latencys[dec_seq_id][iter_id] = request_latency

        average_tpot = np.mean(result_e2e_dec_latencys) / (
            dec_steps * num_seq_to_simulate
        )
        average_seq_latency = average_tpot * dec_steps
        average_request_latency = np.mean(iter_request_latencys)

        assert average_tpot >= latency_s_dec_per_token

        return average_seq_latency, average_request_latency

    def simulate_iterative_retrieval(
        self,
        batch_size_dec: int = 16,
        batch_size_initial: int = 16,
        batch_size_iterative: int = 1,
        latency_s_dec_per_token: int = 10,
        latency_s_retrieval_initial: int = 10,
        latency_s_prefill_initial: int = 10,
        latency_s_retrieval_iterative: int = 1,
        latency_s_prefill_iterative: int = 1,
        dec_steps: int = 256,
        retrieval_interval: int = 64,
        num_seq_to_simulate: int = 10,
        n_simulation_runs: int = 10,
    ):
        """
        Run the iterative retrieval simulation for multiple runs and return average latencies.
        """
        average_seq_latency_list = []
        average_request_latency_list = []
        for _ in range(n_simulation_runs):
            average_seq_latency, average_request_latency = (
                self.simulate_iterative_retrieval_one_run(
                    batch_size_dec=batch_size_dec,
                    batch_size_initial=batch_size_initial,
                    batch_size_iterative=batch_size_iterative,
                    latency_s_dec_per_token=latency_s_dec_per_token,
                    latency_s_retrieval_initial=latency_s_retrieval_initial,
                    latency_s_prefill_initial=latency_s_prefill_initial,
                    latency_s_retrieval_iterative=latency_s_retrieval_iterative,
                    latency_s_prefill_iterative=latency_s_prefill_iterative,
                    dec_steps=dec_steps,
                    retrieval_interval=retrieval_interval,
                    num_seq_to_simulate=num_seq_to_simulate,
                )
            )
            average_seq_latency_list.append(average_seq_latency)
            average_request_latency_list.append(average_request_latency)

        average_seq_latency = np.mean(average_seq_latency_list)
        average_request_latency = np.mean(average_request_latency_list)

        return average_seq_latency, average_request_latency

    def get_iterative_result_df_columns(self):
        columns = []
        columns += ["latency_s", "latency_s_ttft", "latency_s_tpot"]
        columns += ["qps_per_chip", "qps"]
        columns += ["batch_size_request", "batch_size_iterative", "batch_size_decode"]
        columns += ["num_chips_e2e"]
        columns += [
            f"num_chips_{stage}" for stage in self.stages if stage != "retrieval"
        ]
        columns += ["num_retrieval_servers"]
        columns += ["placement_policy", "scheduling_policy"]
        return columns

    def assemble_cost_iterative_retrieval(
        self,
        physical_mapping: PhysicalMapping = PhysicalMapping(),
        scheduling_policy: str = "continuous-batching",
        max_batch_size_request: int = 32,
        no_microbatching: bool = True,
        retrieval_interval: int | None = None,
        possible_batch_sizes_request: list[int] | None = None,
        possible_batch_sizes_iterative: list[int] | None = None,
        possible_batch_sizes_decode: list[int] | None = None,
        neglect_retrieval_latency: bool = False,
        neglect_prefill_latency: bool = False,
    ):
        """Assemble the cost of one specific combination of performance pareto,
        mapping_policy, and scheduling_policy.
        """
        print(
            "WARNING: Event-based simulation deprecated, please consider using the other iterative retrieval performance model function instead."
        )

        placement_policy = physical_mapping.placement_policy
        assert placement_policy in ["disaggregated"]
        assert scheduling_policy in ["continuous-batching"]
        assert no_microbatching, "No microbatching allowed for the sake of performance"
        assert self.stages == ["retrieval", "prefill", "decode"]
        if retrieval_interval is None:
            retrieval_interval = self.retrieval_interval

        # Define the result dataframe
        columns = self.get_iterative_result_df_columns()

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
        all_placement_strategies = [[[stage] for stage in self.stages]]
        print("All valid collocation strategies: ", all_placement_strategies)

        time_start = time.time()
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
                # print("stage: ", stage, "possible_batch_sizes_dict: ", possible_batch_sizes_dict[stage])
            if possible_batch_sizes_decode is not None:
                possible_batch_sizes_dict["decode"] = possible_batch_sizes_decode

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
                # print("stage: ", stage, "latency_table_dict: ", latency_table_dict[stage])
            if neglect_prefill_latency:
                for batch_size in latency_table_dict["prefill"]:
                    latency_table_dict["prefill"][batch_size] = 0
            if neglect_retrieval_latency:
                for batch_size in latency_table_dict["retrieval"]:
                    latency_table_dict["retrieval"][batch_size] = 0

            # No microbatching: all stages before decode use the same batch size
            if possible_batch_sizes_request is None:
                possible_batch_sizes_request = get_power_of_two_list(
                    max_batch_size_request
                )
            else:
                possible_batch_sizes_request = [
                    i
                    for i in possible_batch_sizes_request
                    if i <= max_batch_size_request
                ]

            new_rows = []
            for batch_size_request in possible_batch_sizes_request:

                # Retrieval
                latency_s_retrieval_init = latency_table_dict["retrieval"][
                    batch_size_request
                ]

                finish_time_retrieval = self.get_pipeline_latency_table(
                    previous_stage_finish_time_LUT=None,
                    batch_size_request=max_batch_size_request,
                    batch_size_current_stage=batch_size_request,
                    latency_per_batch_current_stage=latency_s_retrieval_init,
                )

                # prefill
                latency_s_prefill_init = latency_table_dict["prefill"][batch_size_request]

                finish_time_prefill = self.get_pipeline_latency_table(
                    previous_stage_finish_time_LUT=finish_time_retrieval,
                    batch_size_request=max_batch_size_request,
                    batch_size_current_stage=batch_size_request,
                    latency_per_batch_current_stage=latency_s_prefill_init,
                )
                # print("latency_s_retrieval_init: ", latency_s_retrieval_init)
                # print("latency_s_prefill_init: ", latency_s_prefill_init)
                # print("finish_time_prefill: ", finish_time_prefill)

                latency_s_ttft = np.average(
                    [finish_time_prefill[i] for i in range(batch_size_request)]
                )

                # This setup assumes flexible retrieval interval
                for batch_size_decode in possible_batch_sizes_dict["decode"]:
                    if batch_size_decode < batch_size_request:
                        continue

                    latency_s_decode = latency_table_dict["decode"][batch_size_decode]
                    latency_s_decode_per_token = latency_s_decode / self.dec_steps

                    if possible_batch_sizes_iterative is None:
                        batch_size_iterative_list = get_power_of_two_list(
                            batch_size_request
                        )
                    else:
                        batch_size_iterative_list = [
                            i
                            for i in possible_batch_sizes_iterative
                            if i <= batch_size_request
                        ]

                    for batch_size_iterative in batch_size_iterative_list:
                        print(
                            f"batch_size_request: {batch_size_request}\t"
                            + f"batch_size_decode: {batch_size_decode}\t"
                            + f"batch_size_iterative: {batch_size_iterative}"
                        )

                        latency_s_retrieval_iterative = latency_table_dict["retrieval"][
                            batch_size_iterative
                        ]
                        latency_s_prefill_iterative = latency_table_dict["prefill"][
                            batch_size_iterative
                        ]

                        average_seq_latency, average_request_latency = (
                            self.simulate_iterative_retrieval(
                                batch_size_dec=batch_size_decode,
                                batch_size_initial=batch_size_request,
                                batch_size_iterative=batch_size_iterative,
                                latency_s_dec_per_token=latency_s_decode_per_token,
                                latency_s_retrieval_initial=latency_s_retrieval_init,
                                latency_s_prefill_initial=latency_s_prefill_init,
                                latency_s_retrieval_iterative=latency_s_retrieval_iterative,
                                latency_s_prefill_iterative=latency_s_prefill_iterative,
                                dec_steps=self.dec_steps,
                                retrieval_interval=retrieval_interval,
                                num_seq_to_simulate=10,  # the simulation can go beyond the len of one sequence
                                n_simulation_runs=10,
                            )
                        )
                        assert average_seq_latency > latency_s_decode
                        latency_s_e2e = average_seq_latency + latency_s_ttft
                        latency_s_tpot = average_seq_latency / self.dec_steps
                        qps = batch_size_decode / average_seq_latency
                        qps_per_chip = qps / num_chips_e2e

                        # Append the result row to num_chips_e2e
                        new_rows.append(
                            {
                                "latency_s": latency_s_e2e,
                                "latency_s_ttft": latency_s_ttft,
                                "latency_s_tpot": latency_s_tpot,
                                "qps": qps,
                                "qps_per_chip": qps_per_chip,
                                "batch_size_request": batch_size_request,
                                "batch_size_iterative": batch_size_iterative,
                                "batch_size_decode": batch_size_decode,
                                "num_chips_e2e": num_chips["e2e"],
                                "num_chips_prefill": num_chips["prefill"],
                                "num_chips_decode": num_chips["decode"],
                                "num_retrieval_servers": num_retrieval_servers,
                                "placement_policy": placement_policy,
                                "scheduling_policy": scheduling_policy,
                            }
                        )

            print(f"logic time: {time.time() - time_start}")
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
