"""
For each stage of RAG, get the performance across systems, models, and batch sizes.

The user can tune the configuration of the sweeps defined in the following functions:
    get_perf_main_llm()
    get_perf_rewriter()
    get_perf_encoder()
    get_perf_reranker()

A few reference files or folders in GenZ:

genz/tests/test_decode_time.py -> decode functionalities
genz/tests/test_prefill_time.py -> prefill functionalities
genz/Systems/system_configs.py -> supported hardware
genz/GenZ/Models/Model_sets/meta.py -> supported models
"""

import pandas as pd
import numpy as np

from GenZ import prefill_moddeling, decode_moddeling, get_configs, System, get_AR_time
from GenZ.Models.default_models import ModelConfig


def is_power_of_two(n):
    """Helper function to check if a number is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def get_powers_of_two_up_to(n):
    """Return a list of powers of two up to the given number n."""
    powers_of_two = []
    power = 1
    while power <= n:
        powers_of_two.append(power)
        power *= 2
    return powers_of_two


def get_parallelism(num_devices: int) -> list[tuple[int, int]]:
    """
    Get all possible parallelism combinations for a given number of devices.
    """
    # Check if num_devices is a power of two
    if not is_power_of_two(num_devices):
        raise ValueError(f"{num_devices} is not a power of two.")

    parallelisms = []

    # Iterate through all possible powers of two (1, 2, 4, ..., num_devices)
    tp = 1
    while tp <= num_devices:
        pp = num_devices // tp
        if is_power_of_two(pp):
            parallelisms.append((tp, pp))
        tp *= 2  # Increase tp by a power of two

    return parallelisms


def get_pareto_df(df):
    """
    Remove the rows with larger batch sizes but not better performance.
    Specifically, clean the DataFrame by removing rows where the latency for
        a batch size of 2n is exactly twice the latency for a batch size of n.
    """
    rows_to_drop = []

    # List of columns to group by, excluding 'batch_size' and 'time_ms'
    group_columns = [
        "model.name",
        "model.stage",
        "model.dec_steps",
        "model.seq_len_inference_prefill",
        "hardware.name",
        "hardware.num_chips",
    ]

    # Iterate through the unique groups (excluding batch_size and time_ms)
    for group, group_df in df.groupby(group_columns):
        batch_sizes = sorted(group_df["batch_size"].unique())

        for batch_size in batch_sizes:
            if batch_size == batch_sizes[-1]:
                break

            # Check if the performance scaling is still there
            n_row = group_df[group_df["batch_size"] == batch_size]
            two_n_row = group_df[group_df["batch_size"] == batch_size * 2]
            latency_n = n_row["time_ms"].values[0]
            latency_2n = two_n_row["time_ms"].values[0]
            # Achieve 99% of the performance with smaller batch sizes
            rtol = 1e-02
            if latency_2n >= 2 * latency_n or np.allclose(
                latency_2n, 2 * latency_n, rtol=rtol
            ):
                # drop all the rows of batch size >= 2n
                rows_to_drop.extend(
                    group_df[group_df["batch_size"] >= batch_size * 2].index
                )
                break

    # Drop the rows from the DataFrame
    df.drop(rows_to_drop, inplace=True)

    return df


def get_perf_list(
    model_list=["llama2_7b", "llama2_70b"],
    system_list=["TPUv5p", "H100_GPU", "B100"],
    num_device_list=get_powers_of_two_up_to(16),
    stage="decode",
    input_tokens=512,
    output_tokens=256,
    beam=1,
    bits="int8",
):
    """
    Output CSV headers:
    model.name,model.stage,model.dec_steps,model.seq_len_inference_prefill,hardware.name,hardware.num_chips,batch_size,time_ms
    """
    assert stage in ["decode", "prefill"]
    result_list = []

    for model in model_list:
        for system in system_list:
            for num_devices in num_device_list:
                # Get the parallelism combinations for the current number of devices
                parallelism_combinations = get_parallelism(num_devices)

                for tp, pp in parallelism_combinations:
                    # Try various batch sizes
                    batch_size = 1
                    while True:
                        try:
                            if stage == "decode":
                                # Generate the current result
                                decode_output = decode_moddeling(
                                    model=model,
                                    batch_size=batch_size,
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                    Bb=beam,
                                    system_name=system,
                                    bits=bits,
                                    tensor_parallel=tp,
                                    pipeline_parallel=pp,
                                    debug=False,
                                )

                                # Append the result
                                result_list.append(
                                    {
                                        "model.name": model,
                                        "model.stage": "decode",
                                        "model.dec_steps": output_tokens,
                                        "model.seq_len_inference_prefill": input_tokens,
                                        "hardware.name": system,
                                        "hardware.num_chips": num_devices,
                                        "batch_size": batch_size,
                                        "time_ms": decode_output["Latency"],
                                    }
                                )

                            elif stage == "prefill":
                                # Generate the current result
                                prefill_output = prefill_moddeling(
                                    model=model,
                                    batch_size=batch_size,
                                    input_tokens=input_tokens,
                                    system_name=system,
                                    bits=bits,
                                    tensor_parallel=tp,
                                    pipeline_parallel=pp,
                                    debug=False,
                                )

                                # Append the result
                                result_list.append(
                                    {
                                        "model.name": model,
                                        "model.stage": "prefill",
                                        "model.dec_steps": 1,
                                        "model.seq_len_inference_prefill": input_tokens,
                                        "hardware.name": system,
                                        "hardware.num_chips": num_devices,
                                        "batch_size": batch_size,
                                        "time_ms": prefill_output["Latency"],
                                    }
                                )

                            batch_size *= 2  # Increase batch size for next iteration

                        except Exception as e:
                            # If there is system device memory overflow, stop increase batch size
                            break

    df = pd.DataFrame(result_list)

    # Only keep the best performance for each combination of model, system, and batch size
    df_optimal = df.loc[
        df.groupby(
            [
                "model.name",
                "model.stage",
                "model.dec_steps",
                "model.seq_len_inference_prefill",
                "hardware.name",
                "hardware.num_chips",
                "batch_size",
            ]
        ).time_ms.idxmin()
    ]
    # Only keep the smallest batch size when performance improvement stops
    df_optimal = get_pareto_df(df_optimal)

    return df_optimal


"""
User-defined sweep parameters starts:
"""


def get_perf_main_llm():

    model_list = ["llama2_7b", "llama2_70b"]
    system_list = ["TPUv5p", "H100_GPU", "B100"]
    num_device_list = get_powers_of_two_up_to(4)
    input_tokens = 512
    output_tokens = 256
    beam = 1
    bits = "int8"

    df_results = []
    for stages in ["prefill", "decode"]:
        df = get_perf_list(
            model_list=model_list,
            system_list=system_list,
            num_device_list=num_device_list,
            stage=stages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            beam=beam,
            bits=bits,
        )
        df_results.append(df)

    # Concatenate the DataFrames
    df_combined = pd.concat(df_results, ignore_index=True)
    output_file_path = "perf_results/main_llm_perf.csv"
    df_combined.to_csv(output_file_path, index=False)
    print(f"DataFrame successfully written to {output_file_path}")


def get_perf_rewriter():

    model_list = ["llama2_7b"]
    system_list = ["TPUv5p", "H100_GPU", "B100"]
    num_device_list = get_powers_of_two_up_to(4)
    input_tokens = 32
    output_tokens = 32
    beam = 1
    bits = "int8"

    df_results = []
    for stages in ["prefill", "decode"]:
        df = get_perf_list(
            model_list=model_list,
            system_list=system_list,
            num_device_list=num_device_list,
            stage=stages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            beam=beam,
            bits=bits,
        )
        df_results.append(df)

    # Concatenate the DataFrames
    df_combined = pd.concat(df_results, ignore_index=True)
    output_file_path = "perf_results/rewriter_perf.csv"
    df_combined.to_csv(output_file_path, index=False)
    print(f"DataFrame successfully written to {output_file_path}")


def get_perf_encoder():
    model_name = "sbert_120M"
    sbert_120m_config = ModelConfig(
        model=model_name,
        hidden_size=384,
        num_attention_heads=12,
        num_ffi=2,
        intermediate_size=1536,
        num_decoder_layers=12,
        vocab_size=50272,
        hidden_act="relu",
        max_model_len=2 * 1024,
    )

    model_list = [sbert_120m_config]
    system_list = ["TPUv5p", "H100_GPU", "B100"]
    num_device_list = get_powers_of_two_up_to(4)
    input_tokens = 128
    output_tokens = 1
    bits = "bf16"
    df_results = get_perf_list(
        model_list=model_list,
        system_list=system_list,
        num_device_list=num_device_list,
        stage="prefill",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        bits=bits,
    )
    # Self defined model will end up with verbose name
    # Replace model.name with model.name.model
    df_results["model.name"] = model_name
    # Concatenate the DataFrames
    output_file_path = "perf_results/encoder_perf.csv"
    df_results.to_csv(output_file_path, index=False)
    print(f"DataFrame successfully written to {output_file_path}")


def get_perf_reranker():

    model_name = "sbert_120M"
    sbert_120m_config = ModelConfig(
        model=model_name,
        hidden_size=384,
        num_attention_heads=12,
        num_ffi=2,
        intermediate_size=1536,
        num_decoder_layers=12,
        vocab_size=50272,
        hidden_act="relu",
        max_model_len=2 * 1024,
    )

    model_list = [sbert_120m_config]
    system_list = ["TPUv5p", "H100_GPU", "B100"]
    num_device_list = get_powers_of_two_up_to(4)
    input_tokens = 128
    output_tokens = 1
    bits = "bf16"
    df_results = get_perf_list(
        model_list=model_list,
        system_list=system_list,
        num_device_list=num_device_list,
        stage="prefill",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        bits=bits,
    )
    # Self defined model will end up with verbose name
    # Replace model.name with model.name.model
    df_results["model.name"] = model_name
    # Concatenate the DataFrames
    output_file_path = "perf_results/reranker_perf.csv"
    df_results.to_csv(output_file_path, index=False)
    print(f"DataFrame successfully written to {output_file_path}")


if __name__ == "__main__":

    # Get the performance of each stage of RAG
    get_perf_main_llm()
    get_perf_rewriter()
    get_perf_encoder()
    get_perf_reranker()
