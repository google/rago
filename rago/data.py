import copy
import os
import pandas as pd

from typing import Any, Dict

from utils import is_power_of_two, get_pareto_per_chip_number


def load_csv(file_path: str):
    """
    Function to load inference / retrieval performance data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def load_performance_data(
    file_path: str,
    data_type: str = "inference",
):
    """
    Function to load performance data from a CSV file based on the data type ('retrieval' or 'inference').
    Filters out unnecessary columns based on the specified data type.

    Parameters:
    file_path (str): Path to the CSV file.
    data_type (str): The type of data ('retrieval' or 'inference').

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """

    df = load_csv(file_path)

    if df is None:
        raise ValueError("Failed to load data. Please check the file path and format.")

    # Define the columns to keep for inference
    inference_columns = [
        "model.name",
        "model.stage",
        "model.dec_steps",
        "model.seq_len_inference_prefill",
        "hardware.name",
        "hardware.num_chips",
        "batch_size",
        "time_ms",
    ]

    # Define the columns to keep for retrieval
    retrieval_columns = [
        "db.name",
        "hardware.name",
        "hardware.num_servers",
        "batch_size",
        "time_ms",
    ]

    # Filter the columns based on the data type
    if data_type == "inference":
        # Check if the necessary columns exist in the DataFrame
        missing_columns = [col for col in inference_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns for inference data: {missing_columns}")
        # Keep only the relevant columns for inference
        df = df[inference_columns]

    elif data_type == "retrieval":
        # Check if the necessary columns exist in the DataFrame
        missing_columns = [col for col in retrieval_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns for retrieval data: {missing_columns}")
        # Keep only the relevant columns for retrieval
        df = df[retrieval_columns]

    else:
        raise ValueError(
            "Unsupported data type. Please use 'inference' or 'retrieval'."
        )

    return df


def preprocess_data(
    df: pd.DataFrame,
    return_power_of_two: bool = True,
    stage: str = "prefill",
    num_chips_per_server: int | None = 4,
    return_pareto: bool = True,
):
    """
    Preprocesses performance data based on the given conditions:

    1. Filters based on power-of-two batch sizes, num chips, and num servers.
    2. Filters based on the stage (prefill, decode, or retrieval).
    3. Calculates latency_s for different stages.
    4. Calculates QPS and QPS per chip.
    5. Filters the Pareto frontier based on latency and QPS/chip if return_pareto is True.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the performance data.
    return_power_of_two (bool): Whether to return only rows where batch_size, num_chips, and num_servers are powers of two.
    stage (str): The stage to filter for ('prefill', 'decode', or 'retrieval').
    num_chips_per_server (int or None): The number of chips per server (defaults to 4). If None, ignores this filter.
    return_pareto (bool): Whether to filter for the Pareto frontier based on latency and QPS per chip.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """
    df = copy.deepcopy(df)  # Avoid modifying the original DataFrame
    assert stage in [
        "prefill",
        "decode",
        "retrieval",
    ], "Stage must be one of 'prefill', 'decode', or 'retrieval'."

    # chip to server ratio for retrieval stage
    if stage == "retrieval":
        assert num_chips_per_server is not None
        df["hardware.num_chips"] = df["hardware.num_servers"] * num_chips_per_server

    # Step 1: Filter based on return_power_of_two
    if return_power_of_two:
        df = df[df["batch_size"].apply(is_power_of_two)]
        if stage in ["prefill", "decode"]:
            df = df[df["hardware.num_chips"].apply(is_power_of_two)]
        else:
            df = df[df["hardware.num_servers"].apply(is_power_of_two)]

    # Step 2: Filter based on the stage
    if stage in ["prefill", "decode"]:
        df = df[df["model.stage"] == stage]

    # Step 3: Calculate latency_s
    if stage == "decode":
        df["latency_s"] = df["model.dec_steps"] * df["time_ms"] / 1000
    else:
        df["latency_s"] = df["time_ms"] / 1000
    # Remove the original time_ms column
    df = df.drop(columns=["time_ms"])

    # Step 4: Calculate QPS and QPS per chip
    df["qps"] = df["batch_size"] / df["latency_s"]  # QPS = 1 / latency in seconds
    df["qps_per_chip"] = df["qps"] / df["hardware.num_chips"]

    # Step 5: Filter out the Pareto frontier based on latency and QPS per chip
    if return_pareto:
        df = get_pareto_per_chip_number(
            df,
            stage,
            unique_groups_retrieval=["db.name"],
            unique_groups_inference=[
                "model.name",
                "model.seq_len_inference_prefill",
                "model.dec_steps",
            ],
        )
    # Remove naming prefill
    df.columns = df.columns.str.replace("hardware.", "", regex=True)
    if stage == "retrieval":
        df.columns = df.columns.str.replace("db.", "", regex=True)
    else:
        df.columns = df.columns.str.replace("model.", "", regex=True)

    return df


if __name__ == "__main__":
    # Define relative paths to the CSV files
    inference_file_path = os.path.abspath("../data/dummy/example_inference_data.csv")
    retrieval_file_path = os.path.abspath("../data/dummy/example_retrieval_data.csv")

    # Load inference data
    try:
        inference_data = load_performance_data(
            inference_file_path, data_type="inference"
        )
        print("Inference Data Loaded Successfully. First few rows:")
        print(inference_data.head())  # Display the first few rows to verify

        # Process the inference data using preprocess_data
        for stage in ["prefill", "decode"]:
            processed_inference_data = preprocess_data(
                inference_data, stage=stage, return_pareto=True
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
        processed_retrieval_data = preprocess_data(
            retrieval_data, stage="retrieval", return_pareto=True
        )
        print("\nProcessed Retrieval Data:")
        print(
            processed_retrieval_data
        )  # Display the first few rows of the processed data

    except ValueError as e:
        print(f"Error loading retrieval data: {e}")
