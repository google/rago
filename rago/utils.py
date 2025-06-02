import copy
import pandas as pd
import numpy as np

from typing import Any, Dict


def get_filtered_df(df: pd.DataFrame, constraints: Dict[str, Any] = {}) -> pd.DataFrame:
    """
    Given a DataFrame `df` and a dictionary of constraints, return the selected rows based on those constraints.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    constraints (dict[str, Any]): A dictionary where keys are column names and values are the constraint values.
                                 The values can either be a single value or a list of values to filter on.
                                 If a value is `None`, that constraint is ignored.

    Returns:
    pd.DataFrame: The filtered DataFrame containing only rows that match the given constraints.

    The constraints dictionary works as follows:
    - If the value for a column is a list, it selects rows where the column values are in that list.
    - If the value is a single value, it selects rows where the column value equals that value.
    """
    # Create a deep copy of the original DataFrame to avoid modifying the input dataframe
    df_selected = copy.deepcopy(df)

    # Loop through the constraints dictionary and apply filters
    if constraints is not None:
        for key, value in constraints.items():
            # Skip if the value for the key is None
            if value is None:
                continue

            # If the value is a list, filter the DataFrame using `isin()`
            if isinstance(value, list):
                df_selected = df_selected.loc[df_selected[key].isin(value)]
            # If the value is a single value, filter the DataFrame for equality
            else:
                df_selected = df_selected.loc[df_selected[key] == value]

    return df_selected


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Get the Pareto frontier mask, identifying the Pareto-efficient points.

    Reference: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    Args:
        costs (np.ndarray): An (n_points, n_costs) array where each row corresponds to a point,
                             and each column corresponds to a cost. Always lower is better for each cost.

    Returns:
        np.ndarray: A boolean array where `True` indicates a Pareto-efficient point.
    """
    n_points = costs.shape[0]
    is_efficient_mask = np.ones(
        n_points, dtype=bool
    )  # Initialize all points as efficient

    # Iterate over each point
    for i in range(n_points):
        # Compare point `i` with all subsequent points
        # is_dominated = np.any(np.all(costs[is_efficient_mask] < costs[i], axis=1))
        is_dominated = np.any(np.all(costs[is_efficient_mask] <= costs[i], axis=1) & np.any(costs[is_efficient_mask] < costs[i], axis=1))

        if is_dominated:
            is_efficient_mask[i] = False  # Mark as dominated

    return is_efficient_mask


def get_pareto_df(
    sweep_df: pd.DataFrame,
    keys_lower_is_better: list[tuple[str, bool]],
):
    """
    Get the Pareto optimal DataFrame, based on the specified performance attributes.

    Args:
        sweep_df: The dataframe containing the performance data.
        keys_lower_is_better: A list of tuples where each tuple contains:
            - a performance attribute name (str),
            - a boolean indicating whether lower values are better for that attribute.

    Returns:
        A DataFrame containing the Pareto-optimal points.
    """
    cost_arrays = []

    # Iterate over the performance columns to prepare cost arrays
    for key, lower_is_better in keys_lower_is_better:
        cost = sweep_df[key].to_numpy().reshape((-1, 1))

        # If the performance attribute is such that lower values are not better, invert the cost
        if not lower_is_better:
            cost = 1 / cost

        cost_arrays.append(cost)

    # Concatenate the cost arrays
    cost_arrays = np.concatenate(cost_arrays, axis=1)

    # Determine the Pareto efficient points
    mask = is_pareto_efficient(cost_arrays)

    # Get the Pareto-efficient points from the original dataframe
    performance_pareto = sweep_df.loc[mask]

    # Sort by the first key
    key, lower_is_better = keys_lower_is_better[0]
    performance_pareto = performance_pareto.sort_values(
        by=key, ascending=lower_is_better
    )

    return performance_pareto


def get_pareto_per_chip_number(
    df: pd.DataFrame,
    stage: str,
    unique_groups_retrieval: list[str] = ["db.name"],
    unique_groups_inference: list[str] = [
        "model.name",
        "model.seq_len_inference_prefill",
        "model.dec_steps",
    ],
) -> pd.DataFrame:
    """
    Return the Pareto-optimal points for each chip number, grouped by unique performance categories.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the performance data.
    stage (str): The stage for filtering ('prefill', 'decode', or 'retrieval').
    unique_groups_retrieval (list): List of columns to define unique groups for retrieval.
    unique_groups_inference (list): List of columns to define unique groups for inference.

    Returns:
    pd.DataFrame: The DataFrame filtered to include only Pareto-optimal points.
    """
    # Define unique group columns based on the stage
    unique_group_cols = (
        unique_groups_retrieval if stage == "retrieval" else unique_groups_inference
    )

    # Ensure that the columns exist in the DataFrame
    unique_group_cols = [col for col in unique_group_cols if col in df.columns]
    print(unique_group_cols)
    unique_groups = set(df[unique_group_cols].itertuples(index=False, name=None))
    print(unique_groups)

    # Dictionary to store Pareto-optimal data per group
    df_pareto_dict = {}

    # Iterate through unique groups and calculate the Pareto frontier for each
    for group in unique_groups:
        selected_df = df.copy()

        # Filter data based on the unique group values
        for col_name, group_value in zip(unique_group_cols, group):
            selected_df = selected_df[selected_df[col_name] == group_value]

        # Get a sorted list of unique chip numbers
        num_chips_list = sorted(set(selected_df["hardware.num_chips"]))

        # Iterate over chip numbers and calculate the Pareto frontier
        df_pareto_per_group_dict = {}
        for num_chips in num_chips_list:
            assert is_power_of_two(
                num_chips
            ), f"Num chips must be a power of two. Found: {num_chips}"

            # Filter the data for the current chip number
            df_subset = selected_df[selected_df["hardware.num_chips"] == num_chips]

            # Adjust data for half the chips if necessary
            num_chips_half = num_chips // 2
            if num_chips_half in num_chips_list:
                df_pareto_alter = df_pareto.copy()
                df_pareto_alter["batch_size"] *= 2
                df_pareto_alter["qps"] *= 2
                df_pareto = pd.concat([df_pareto, df_pareto_alter], axis=0)

            # Get Pareto frontier for the current subset
            df_pareto = get_pareto_df(
                df_subset,
                keys_lower_is_better=[
                    ("latency_s", True),
                    ("qps_per_chip", False),
                    ("qps", False),
                ],
            )

            # Store Pareto data for this chip number
            df_pareto_per_group_dict[num_chips] = df_pareto

        # Concatenate Pareto data for all chip numbers within this group
        df_pareto_dict[group] = pd.concat(df_pareto_per_group_dict.values())

    # Concatenate all groups' Pareto data into the final DataFrame
    df_pareto_final = pd.concat(df_pareto_dict.values())
    return df_pareto_final


def print_df_key_performance(
    sweep_df: pd.DataFrame,
    keys_lower_is_better: list[tuple[str, bool]] = [
        ("latency_s", True),
        ("qps_per_chip", False),
        ("qps", False),
    ],
    print_rows: bool = True,
):
    """
    Given a dataframe of sweep results, print the key rows of performance numbers
    """
    for perf_key, lower_is_better in keys_lower_is_better:
        # Get the min, max rows
        if lower_is_better:
            best_row = sweep_df.loc[sweep_df[perf_key].idxmin()]
            worst_row = sweep_df.loc[sweep_df[perf_key].idxmax()]
        else:
            best_row = sweep_df.loc[sweep_df[perf_key].idxmax()]
            worst_row = sweep_df.loc[sweep_df[perf_key].idxmin()]
        print(
            f"Best {perf_key}: {best_row[perf_key]}" + 
            (f"\n{best_row}" if print_rows else "")
        )
        print(
            f"Worst {perf_key}: {worst_row[perf_key]}" + 
            (f"\n{worst_row}" if print_rows else "")
        )


def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two.

    Returns:
        bool: True if `n` is a power of two, False otherwise.
    """
    return (n > 0) and (n & (n - 1)) == 0


def get_power_of_two_list(max_num: int, start: int = 1) -> list[int]:
    """
    Generate a list of powers of two, starting from `start`, up to `max_num`.

    Args:
        max_num (int): The upper limit for the powers of two.
        start (int): The starting value (default is 1).

    Returns:
        list[int]: A list of powers of two from `start` to `max_num`.

    Example:
        get_power_of_two_list(32) -> [1, 2, 4, 8, 16, 32]
    """
    possible_nums = []
    num = start

    while num <= max_num:
        possible_nums.append(num)
        num *= 2

    return possible_nums
