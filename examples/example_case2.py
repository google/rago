"""
Run performance evaluateion of RAG for long-context processing

Dataflow: Database Encode -> Retrieval -> LLM Prefill -> LLM Decode

Use can config the parameters at the beginning of this file to specify the runs.
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
encoder_file_path = os.path.abspath("../data/llm_perf_sim_genz/encoder_perf.csv")
inference_file_path = os.path.abspath("../data/llm_perf_sim_genz/main_llm_perf.csv")
# encoder_file_path = os.path.abspath("../data/dummy/example_encoder_data.csv")
# inference_file_path = os.path.abspath("../data/dummy/example_inference_data.csv")
# retrieval_file_path = os.path.abspath("../data/dummy/example_retrieval_data.csv")
retrieval_file_path = os.path.abspath("../data/retrieval_perf_sim/retrieval_perf.csv")

print(
    "Note: the retrieval performance here must be based on the assumption"
    " that there are multiple individual databases (no bandwidth sharing when batching)"
)

# The following names can be None, in which case the df filtering does not apply
encoder_model_name = "sbert_120M"
encoder_hardware_name = "H100_GPU"
inference_model_name = "llama2_70b"
inference_hardware_name = "H100_GPU"
# encoder_model_name = "st100m"
# encoder_hardware_name = "xpu"
# inference_model_name = "llama70b"
# inference_hardware_name = "xpu"
retrieval_db_name = "small_100k"
retrieval_hardware_name = "cpu_dram"

# Which functions to execute
run_single_test = True  # Set to True to run cost modeling on a single system
run_sweep_across_systems = True  # Set to True to run the sweep across systems

# System, model, and sweep configurations
encode_seq_len_inference_prefill = 128  # database encoder sequence lengths
encode_db_nvec = 100 * 1000  # number of passages to encode == vectors in the database
seq_len_inference_prefill = 512  # prefill len
dec_steps = 256  # decode len
num_chips_per_server = 4  # chip/server ratio
max_batch_size_request = 1024  # max allowed batch sizes to evaluate
placement_policy = "disaggregated"  # 'collocated' or 'disaggregated'

# for single hardware test, specify the system to evaluate
num_retrieval_servers = 32
num_chips_encode = 16
num_chips_prefill = 8
num_chips_decode = 8

# for sweep across systems, specify the range
min_num_chips = 32
max_num_chips = 64
max_num_chips_per_stage = 64

# plotting configs
show_plot = False  # Set to True to show the plot
save_plot = True
save_path_all_points = "./img/example_case_2_all.png"  # Path to save the plot
save_path_pareto = "./img/example_case_2_pareto.png"  # Path to save the plot
save_path_breakdown = (
    "./img/example_case_2_workload_breakdown.png"  # Path to save the plot
)

""" User-configurable parameters ends """

all_sweep_infos = {}


# Load the data for testing
def load_test_data():

    # Load encoder data
    try:
        encoder_data = load_performance_data(encoder_file_path, data_type="inference")
        print("Encoder Data Loaded Successfully. First few rows:")
        print(encoder_data.head())  # Display the first few rows to verify

        # Process the encoder data using preprocess_data
        encoder_data_selected = get_filtered_df(
            encoder_data,
            {
                "model.name": encoder_model_name,
                "hardware.name": encoder_hardware_name,
                "model.stage": "prefill",
            },
        )
        processed_encoder_data = preprocess_data(
            encoder_data_selected, stage="prefill", return_pareto=True
        )
        all_sweep_infos["encoder"] = copy.deepcopy(processed_encoder_data)
        print(f"\nProcessed Encoder Data:")
        print(
            processed_encoder_data
        )  # Display the first few rows of the processed data

    except ValueError as e:
        print(f"Error loading encoder data: {e}")

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

    rag_sweep = RAGSweep(
        encode_db_policy=EncodeDBPolicy(
            run_encode_db=True,
            encode_seq_len_inference_prefill=encode_seq_len_inference_prefill,
            encode_db_nvec=encode_db_nvec,
        ),
        retrieval_policy=RetrievalPolicy(run_retrieval=True, retrieval_pattern="once"),
        stages=["encode", "retrieval", "prefill", "decode"],
        sweep_df={
            "rewrite_prefill": None,
            "rewrite_decode": None,
            "encode": all_sweep_infos["encoder"],
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
    df_single_system_result = rag_sweep.assemble_cost(
        physical_mapping=PhysicalMapping(
            num_retrieval_servers=num_retrieval_servers,
            num_chips={
                "encode": num_chips_encode,
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
        max_batch_size_request=max_batch_size_request,
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
            legend_list=["Case 2"],
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
            legend_list=["Case 2"],
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
            x_tick_labels=["Case 2"],
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
