"""
Run performance evaluation of simple RAG with Iterative Retrieval.

Dataflow: Retrieval -> LLM Prefill -> LLM Decode -> Retrieval -> LLM prefill -> LLM Decode -> ...

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
from rago_sweep_iterative import IterativeRAGSweep
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
# inference_file_path = os.path.abspath("../data/dummy/example_inference_data.csv")
# retrieval_file_path = os.path.abspath("../data/dummy/example_retrieval_data.csv")
retrieval_file_path = os.path.abspath("../data/retrieval_perf_sim/retrieval_perf.csv")

# The following names can be None, in which case the df filtering does not apply
inference_model_name = "llama2_70b"
inference_hardware_name = "H100_GPU"
# inference_model_name = "llama70b"
# inference_hardware_name = "xpu"
retrieval_db_name = "large_64b"
# retrieval_db_name = "large_100b"
retrieval_hardware_name = "cpu_dram"

# Which functions to execute
run_single_test = True  # Set to True to run cost modeling on a single system

# System, model, and sweep configurations
seq_len_inference_prefill = 512  # prefill len
dec_steps = 256  # decode len
num_chips_per_server = 4  # chip/server ratio
max_batch_size_request = 1024  # max allowed batch sizes to evaluate
placement_policy = "disaggregated"  # 'collocated' or 'disaggregated'

# for single hardware test, specify the system to evaluate
num_retrieval_servers = 32
num_chips_prefill = 16
num_chips_decode = 16

# for sweep across systems, specify the range
min_num_chips = 32
max_num_chips = 64
max_num_chips_per_stage = 64

# Iterative retrieval parameters
num_retrievals = 4  # retrievals per sequence
possible_batch_sizes_decode = [1, 4, 16, 64, 256, 1024]
possible_batch_sizes_request = [1, 4, 16, 64]
possible_batch_sizes_iterative = [1, 4, 16, 64]

# plotting configs
show_plot = False  # Set to True to show the plot
save_plot = True
save_path_all_points = "./img/example_case_3_all.png"  # Path to save the plot
save_path_pareto = "./img/example_case_3_pareto.png"  # Path to save the plot

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
    rag_sweep = IterativeRAGSweep(
        encode_db_policy=EncodeDBPolicy(
            run_encode_db=False, encode_seq_len_inference_prefill=None
        ),
        retrieval_policy=RetrievalPolicy(
            run_retrieval=True, retrieval_pattern="iterative"
        ),
        stages=["retrieval", "prefill", "decode"],
        sweep_df={
            "rewrite_prefill": None,
            "rewrite_decode": None,
            "encode": None,
            "retrieval": all_sweep_infos["retrieval"],
            "rerank": None,
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
    if num_retrievals <= 1:
        df_single_system_result = rag_sweep.assemble_cost(
            physical_mapping=PhysicalMapping(
                num_retrieval_servers=num_retrieval_servers,
                num_chips={"prefill": num_chips_prefill, "decode": num_chips_decode},
                placement_policy=placement_policy,
            ),
            scheduling_policy="continuous-batching",
            max_batch_size_request=max_batch_size_request,
        )
    else:
        retrieval_interval = int(dec_steps / num_retrievals)
        df_single_system_result = rag_sweep.assemble_cost_iterative_retrieval(
            physical_mapping=PhysicalMapping(
                num_retrieval_servers=num_retrieval_servers,
                num_chips={"prefill": num_chips_prefill, "decode": num_chips_decode},
                placement_policy=placement_policy,
            ),
            scheduling_policy="continuous-batching",
            max_batch_size_request=max_batch_size_request,
            retrieval_interval=retrieval_interval,
            possible_batch_sizes_request=possible_batch_sizes_request,
            possible_batch_sizes_iterative=possible_batch_sizes_iterative,
            possible_batch_sizes_decode=possible_batch_sizes_decode,
            neglect_retrieval_latency=False,
            neglect_prefill_latency=False,
        )
    # Print the result for the single system
    print("Assembled cost for the specified single system:", df_single_system_result)

    return df_single_system_result, rag_sweep


""" Execute the code """
load_test_data()
if run_single_test:

    print("==== All Performance Points ====")
    df_sweep_results, rag_sweep = test_assemble_cost()
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
    x_key = "latency_s_tpot"  # 'latency_s', 'latency_s_ttft', 'latency_s_tpot', 'qps_per_chip'
    y_key = "qps_per_chip"
    df_pareto = get_pareto_df(
        sweep_df=df_sweep_results,
        keys_lower_is_better=[
            (x_key, get_lower_is_better(x_key)),
            (y_key, get_lower_is_better(y_key)),
        ],
    )
    print(df_pareto)

    if show_plot or save_plot:

        # Plot all result points
        plot_scatter_df(
            df_list=[df_sweep_results],
            legend_list=["Case 3"],
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
            legend_list=["Case 3"],
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
