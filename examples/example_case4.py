"""
Run performance evaluation of a multi-component RAG (Rewrite + Rerank + Retrieval + LLM).

Dataflow: Rewrite -> Retrieval -> LLM Prefill -> LLM Decode -> Rerank
User can configure the parameters at the beginning of this file to specify the runs.
"""

import copy
import gc
import os
import pandas as pd
import sys
import time

# Add the 'rago' directory to sys.path to be able to import it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rago")))

from config import RetrievalPolicy, EncodeDBPolicy, PhysicalMapping
from data import load_performance_data, preprocess_data
from rago_sweep import RAGSweep
from utils import get_filtered_df, get_pareto_df, print_df_key_performance
from plot_utils import (
    get_label,
    get_lower_is_better,
    get_workload_breakdown,
    plot_curve_df,
    plot_scatter_df,
    plot_breakdown,
)

""" User-configurable parameters starts """

# Define relative paths to the CSV files
inference_file_path = os.path.abspath("../data/llm_perf_sim_genz/main_llm_perf.csv")
rewriter_file_path = os.path.abspath("../data/llm_perf_sim_genz/rewriter_perf.csv")
reranker_file_path = os.path.abspath("../data/llm_perf_sim_genz/reranker_perf.csv")
# inference_file_path = os.path.abspath("../data/dummy/example_inference_data.csv")
# rewriter_file_path = os.path.abspath("../data/dummy/example_rewriter_data.csv")
# reranker_file_path = os.path.abspath("../data/dummy/example_reranker_data.csv")
# retrieval_file_path = os.path.abspath("../data/dummy/example_retrieval_data.csv")
retrieval_file_path = os.path.abspath("../data/retrieval_perf_sim/retrieval_perf.csv")

# The following names can be None, in which case the df filtering does not apply
inference_model_name = "llama2_70b"
inference_hardware_name = "H100_GPU"
rewriter_model_name = "llama2_7b"
reranker_model_name = "sbert_120M"
# inference_model_name = "llama70b"
# inference_hardware_name = "xpu"
# rewriter_model_name = "llama8b"
# reranker_model_name = "st100m"
retrieval_db_name = "large_64b"
# retrieval_db_name = "large_100b"
retrieval_hardware_name = "cpu_dram"

# Which functions to execute
run_single_test = True  # Set to True to run cost modeling on a single system
run_sweep_across_systems = True  # Set to True to run the sweep across systems

# System, model, and sweep configurations
seq_len_inference_prefill = 512  # prefill len
dec_steps = 256  # decode len
rewrite_seq_len_inference_prefill = 32  # rewriter prefill len
rewrite_seq_len_prefill_with_template = (
    32  # the prefill is the rewrite instruction template + question
)
rewrite_dec_steps = 32  # rewriter decode len
rerank_seq_len_inference_prefill = 128  # reranker prefill len
rerank_topk = 16  # number of passages to rerank, must be power of two
num_chips_per_server = 4  # chip/server ratio
max_batch_size_request = 1024  # max allowed batch sizes to evaluate
placement_policy = "disaggregated"  # 'collocated' or 'disaggregated'

# for single hardware test, specify the system to evaluate
num_retrieval_servers = 16
num_chips_rewrite_prefill = 1
num_chips_rewrite_decode = 1
num_chips_rerank = 2
num_chips_prefill = 32
num_chips_decode = 32

# for sweep across systems, specify the range
min_num_chips = 16
max_num_chips = 96
max_num_chips_per_stage = 64
possible_num_chips = {
    "rewrite_prefill": [1, 2, 4],
    "rewrite_decode": [1, 2, 4],
    "encode": None,
    "rerank": [1, 2, 4, 32],
    "prefill": [32],
    "decode": [32],
}

# plotting configs
show_plot = False  # Set to True to show the plot
save_plot = True
save_path_all_points = "./img/example_case_4_all.png"  # Path to save the plot
save_path_pareto = "./img/example_case_4_pareto.png"  # Path to save the plot
save_path_breakdown = (
    "./img/example_case_4_workload_breakdown.png"  # Path to save the plot
)

""" User-configurable parameters ends """

all_sweep_infos = {}


# Load the data for testing
def load_test_data():

    # Load inference data
    try:
        inference_data = load_performance_data(
            inference_file_path, data_type="inference"
        )
        print("Inference Data Loaded Successfully. First few rows:")
        print(inference_data.head())  # Display the first few rows to verify

        # Process the inference data using preprocess_data
        for stage in ["prefill", "decode"]:
            inference_data_selected = get_filtered_df(
                inference_data,
                {
                    "model.name": inference_model_name,
                    "hardware.name": inference_hardware_name,
                },
            )
            if stage == "prefill":
                inference_data_selected = get_filtered_df(
                    inference_data_selected,
                    {"model.seq_len_inference_prefill": seq_len_inference_prefill},
                )
            elif stage == "decode":
                inference_data_selected = get_filtered_df(
                    inference_data_selected,
                    {
                        "model.seq_len_inference_prefill": seq_len_inference_prefill,
                        "model.dec_steps": dec_steps,
                    },
                )
            processed_inference_data = preprocess_data(
                inference_data_selected, stage=stage, return_pareto=True
            )
            all_sweep_infos["inference-" + stage] = copy.deepcopy(
                processed_inference_data
            )
            print(f"\nProcessed Inference Data ({stage}):")
            print(
                processed_inference_data
            )  # Display the first few rows of the processed data

    except ValueError as e:
        print(f"Error loading inference data: {e}")

    # Load rewriter data
    try:
        rewriter_data = load_performance_data(rewriter_file_path, data_type="inference")
        print("Rewriter Data Loaded Successfully. First few rows:")
        print(rewriter_data.head())  # Display the first few rows to verify

        # Process the rewriter data using preprocess_data
        for stage in ["prefill", "decode"]:
            rewriter_data_selected = get_filtered_df(
                rewriter_data,
                {
                    "model.name": rewriter_model_name,
                    "hardware.name": inference_hardware_name,
                },
            )
            if stage == "prefill":
                inference_data_selected = get_filtered_df(
                    inference_data_selected,
                    {
                        "model.seq_len_inference_prefill": rewrite_seq_len_inference_prefill
                    },
                )
            elif stage == "decode":
                inference_data_selected = get_filtered_df(
                    inference_data_selected,
                    {
                        "model.seq_len_inference_prefill": rewrite_seq_len_inference_prefill,
                        "model.dec_steps": rewrite_dec_steps,
                    },
                )
            processed_rewriter_data = preprocess_data(
                rewriter_data_selected, stage=stage, return_pareto=True
            )
            all_sweep_infos["rewrite-" + stage] = copy.deepcopy(processed_rewriter_data)
            print(f"\nProcessed Rewriter Data ({stage}):")
            print(
                processed_rewriter_data
            )  # Display the first few rows of the processed data

    except ValueError as e:
        print(f"Error loading rewriter data: {e}")

    # Load reranker data
    try:
        reranker_data = load_performance_data(reranker_file_path, data_type="inference")
        print("Reranker Data Loaded Successfully. First few rows:")
        print(reranker_data.head())  # Display the first few rows to verify

        # Process the reranker data using preprocess_data
        reranker_data_selected = get_filtered_df(
            reranker_data,
            {
                "model.name": reranker_model_name,
                "hardware.name": inference_hardware_name,
                "model.seq_len_inference_prefill": rerank_seq_len_inference_prefill,
            },
        )
        processed_reranker_data = preprocess_data(
            reranker_data_selected, stage="prefill", return_pareto=True
        )
        all_sweep_infos["rerank"] = copy.deepcopy(processed_reranker_data)
        print("\nProcessed Reranker Data:")
        print(
            processed_reranker_data
        )  # Display the first few rows of the processed data

    except ValueError as e:
        print(f"Error loading reranker data: {e}")

    # Load retrieval data
    try:
        retrieval_data = load_performance_data(
            retrieval_file_path, data_type="retrieval"
        )
        print("Retrieval Data Loaded Successfully. First few rows:")
        print(retrieval_data.head())  # Display the first few rows to verify

        # Process the retrieval data using preprocess_data
        retrieval_data_selected = get_filtered_df(
            retrieval_data,
            {"db.name": retrieval_db_name, "hardware.name": retrieval_hardware_name},
        )
        processed_retrieval_data = preprocess_data(
            retrieval_data_selected, stage="retrieval", return_pareto=True
        )
        all_sweep_infos["retrieval"] = copy.deepcopy(processed_retrieval_data)
        print("\nProcessed Retrieval Data:")
        print(
            processed_retrieval_data
        )  # Display the first few rows of the processed data

    except ValueError as e:
        print(f"Error loading retrieval data: {e}")


def init_rago():

    rag_sweep = RAGSweep(
        encode_db_policy=EncodeDBPolicy(run_encode_db=False),
        retrieval_policy=RetrievalPolicy(
            run_retrieval=True,
            retrieval_pattern="once",
            run_rewrite=True,
            rewrite_seq_len_inference_prefill=rewrite_seq_len_inference_prefill,
            rewrite_seq_len_prefill_with_template=rewrite_seq_len_prefill_with_template,
            rewrite_dec_steps=rewrite_dec_steps,
            run_rerank=True,
            rerank_seq_len_inference_prefill=rerank_seq_len_inference_prefill,
            rerank_topk=rerank_topk,
        ),
        stages=[
            "rewrite_prefill",
            "rewrite_decode",
            "retrieval",
            "rerank",
            "prefill",
            "decode",
        ],
        sweep_df={
            "rewrite_prefill": all_sweep_infos["rewrite-prefill"],
            "rewrite_decode": all_sweep_infos["rewrite-decode"],
            "encode": None,
            "retrieval": all_sweep_infos["retrieval"],
            "rerank": all_sweep_infos["rerank"],
            "prefill": all_sweep_infos["inference-prefill"],
            "decode": all_sweep_infos["inference-decode"],
        },
        seq_len_inference_prefill=seq_len_inference_prefill,
        dec_steps=dec_steps,
        num_chips_per_server=num_chips_per_server,
    )

    return rag_sweep


# Test the cost assembler for a specific system
def test_assemble_cost():

    rag_sweep = init_rago()

    # Test assemble_cost with given physical mapping
    df_single_system_result = rag_sweep.assemble_cost(
        physical_mapping=PhysicalMapping(
            num_retrieval_servers=num_retrieval_servers,
            num_chips={
                "rewrite_prefill": num_chips_rewrite_prefill,
                "rewrite_decode": num_chips_rewrite_decode,
                "rerank": num_chips_rerank,
                "prefill": num_chips_prefill,
                "decode": num_chips_decode,
            },
            placement_policy=placement_policy,
        ),
        scheduling_policy="continuous-batching",
        max_batch_size_request=max_batch_size_request,
    )
    print("Assembled cost for the specified single system:", df_single_system_result)


# Test sweep
def test_run_sweep_across_systems():

    rag_sweep = init_rago()

    # Run the sweep test
    df_sweep_results = rag_sweep.run_sweep_across_systems(
        min_num_chips=min_num_chips,
        max_num_chips=max_num_chips,
        max_num_chips_per_stage=max_num_chips_per_stage,
        placement_policy=placement_policy,
        scheduling_policy="continuous-batching",
        max_batch_size_request=max_batch_size_request,  # The prefill here has to be limited to have reasonable runtime
        possible_num_chips=possible_num_chips,
    )
    print("Sweep results across different systems:", df_sweep_results)

    return df_sweep_results, rag_sweep


""" Execute the code """
load_test_data()
if run_single_test:
    test_assemble_cost()

if run_sweep_across_systems:

    print("==== All Performance Points ====")
    df_sweep_results, rag_sweep = test_run_sweep_across_systems()
    print_df_key_performance(
        df_sweep_results,
        keys_lower_is_better=[
            ("latency_s", True),
            ("qps_per_chip", False),
            ("qps", False),
        ],
        print_rows=False,
    )

    print("==== Performance on the Pareto ====")
    x_key = "latency_s_ttft"  # 'latency_s', 'latency_s_ttft', 'latency_s_tpot', 'qps_per_chip'
    y_key = "qps_per_chip"
    df_pareto = get_pareto_df(
        sweep_df=df_sweep_results,
        keys_lower_is_better=[
            (x_key, get_lower_is_better(x_key)),
            (y_key, get_lower_is_better(y_key)),
        ],
    )
    print(df_pareto)

    print("==== Workload Breakdown (based on best QPS/Chip setup) ====")
    workload_distribution = get_workload_breakdown(df_sweep_results, rag_sweep)
    stages = rag_sweep.stages
    for i, stage in enumerate(stages):
        print(f"Stage {stage}: {workload_distribution[i] * 100.0:.2f}% workload")

    if show_plot or save_plot:

        # Plot all result points
        plot_scatter_df(
            df_list=[df_sweep_results],
            legend_list=["Case 4"],
            x_key=x_key,
            y_key=y_key,
            plot_name="All Performance Points",
            x_label=get_label(x_key),
            y_label=get_label(y_key),
            markersize=5,
            label_font=12,
            title_font=12,
            legend_font=10,
            tick_font=11,
            legend_ncol=1,
            legend_loc=None,  # (-0.2, 1.05),
            figsize=(3.2, 1.6),
            plot_style="seaborn-v0_8-colorblind",
            save_path=save_path_all_points,
            show_plot=show_plot,
        )

        # Plot Pareto using a curve
        plot_curve_df(
            df_list=[df_pareto],
            legend_list=["Case 4"],
            color_list=None,
            linestyle_list=None,
            x_key=x_key,
            y_key=y_key,
            plot_name="Pareto Frontier",
            mark_number=False,
            x_label=get_label(x_key),
            y_label=get_label(y_key),
            x_lim=None,
            y_lim=None,
            y_lim_low=None,
            x_lim_low=None,
            markersize=5,
            label_font=12,
            title_font=12,
            legend_font=10,
            tick_font=11,
            legend_ncol=1,
            legend_loc=None,  # (-0.2, 1.05),
            figsize=(3.2, 1.6),
            plot_style="seaborn-v0_8-colorblind",
            #   text="",
            #   text_font=12,
            #   text_loc=text_loc,
            save_path=save_path_pareto,
            show_plot=show_plot,
        )

        # Plot workload breakdown
        plot_breakdown(
            workload_array=[workload_distribution],
            x_tick_labels=["Case 4"],
            legend_list=stages,
            # plot_name=plot_name,
            show_perc=True,
            mark_number=False,
            y_label="Time (%)",
            y_lim=[0, 100],
            label_font=12,
            title_font=12,
            legend_font=10,
            tick_font=11,
            legend_loc=(1.1, -0.05),
            # legend_loc='best',
            figsize=(1.3, 1.6),
            save_path=save_path_breakdown,
            show_plot=show_plot,
        )
