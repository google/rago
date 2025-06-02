import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Any, Dict


def plot_bar(
    latency_array: list[list[float]],  # 2d array: num_experiments x num_components
    x_tick_labels: list[str],  # 1d array: num_experiments
    legend_list: list[str],  # len = num_components
    model_name: str = "",
    mark_number: bool = False,
    y_label: str = "Latency (s)",
    y_log: bool = False,
    save_path: str | None = None,
    show_plot: bool = True,  # whether to show the plot
) -> None:
    """Given different components (e.g., systems), compare their performance
    (y axis in different bars) across experiments (x axis)
    """
    sns.set_theme(style="whitegrid")

    num_experiments = len(x_tick_labels)
    num_components = len(legend_list)

    y_per_component = [[] for _ in range(num_components)]
    for latency_tuple in latency_array:
        assert len(latency_tuple) == num_components
        for i, latency in enumerate(latency_tuple):
            y_per_component[i].append(latency)
    y_per_component = np.array(y_per_component)

    total_width = 0.5
    width = total_width / num_components
    start_pos = -total_width / 2
    _, ax = plt.subplots(1, 1, figsize=(2 * num_experiments, 4))
    x = np.arange(len(x_tick_labels))

    rects = []
    for i in range(num_components):
        rect = ax.bar(
            x + i * width + start_pos, y_per_component[i], width, label=legend_list[i]
        )
        rects.append(rect)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            num = rect.get_height()
            y_pos = rect.get_y() + rect.get_height()
            str_print = f"{num:.2e}"
            ax.annotate(
                str_print,
                xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                xytext=(0, 0),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    if mark_number:
        autolabel(rects)

    ax.set_ylabel(y_label)
    if y_log:
        ax.set_yscale("log")
    ax.set_title(f"{model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels)
    ax.legend()
    if save_path:
        plt.savefig(save_path, transparent=False, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_breakdown(
    workload_array: list[list[float]],  # 2d array: num_experiments x num_components
    x_tick_labels: list[str],  # len = num_experiments
    legend_list: list[str],  # len = num_components
    plot_name: str = "",
    show_perc: bool = False,
    mark_number: bool = False,
    y_label: str | None = None,
    y_lim: list[float] | None = None,
    label_font=12,
    title_font=12,
    legend_font=10,
    tick_font=10,
    legend_loc="best",
    figsize=None,
    save_path: str | None = None,
    show_plot: bool = True,  # whether to show the plot
) -> None:
    """Plots RAG latency breakdown --- with multiple components."""
    sns.set_theme(style="whitegrid")

    num_experiments = len(x_tick_labels)
    num_components = len(legend_list)
    y_per_component = [[] for _ in range(num_components)]
    for workload_tuple in workload_array:
        assert len(workload_tuple) == num_components
        total_latency = np.sum(workload_tuple)
        for i, latency in enumerate(workload_tuple):
            if show_perc:
                y_per_component[i].append(latency / total_latency * 100)
            else:
                y_per_component[i].append(latency)
    y_per_component = np.array(y_per_component)

    width = 0.35
    if not figsize:
        figsize = (2 * num_experiments, 4)
    _, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(x_tick_labels))

    bottom = np.array([0.0] * num_experiments)
    rects = []
    for i in range(num_components):
        if np.any(np.array(y_per_component[i]) > 0):
            rect = ax.bar(
                x, y_per_component[i], width, bottom=bottom, label=legend_list[i]
            )
            rects.append(rect)
        bottom += y_per_component[i]

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            num = rect.get_height()
            y_pos = rect.get_y() + rect.get_height()
            str_print = f"{num:.2e}"
            ax.annotate(
                str_print,
                xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                xytext=(0, 0),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    if mark_number:
        for rect in rects:
            autolabel(rect)

    if y_lim:
        ax.set_ylim(y_lim)
    if y_label:
        ax.set_ylabel(y_label, fontsize=label_font)
    else:
        if show_perc:
            ax.set_ylabel(f"Percentage of latency", fontsize=label_font)
            ax.set_ylim([0, 100])
        else:
            ax.set_ylabel("Latency (s)", fontsize=label_font)
    ax.set_title(plot_name, fontsize=title_font)
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels, fontsize=tick_font)
    ax.legend(loc=legend_loc, fontsize=legend_font)
    if save_path:
        plt.savefig(save_path, transparent=False, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_curve_df(
    df_list: list[pd.DataFrame],  # 1d array: a number of dataframes
    legend_list: list[str],  # len == num_curves
    color_list: list[int] | None = None,
    linestyle_list: list[str] | None = None,
    marker_style_list: list[str] | None = None,
    x_key: str = "latency_s",
    y_key: str = "qps_per_chip",
    plot_name: str = "",
    mark_number: bool = False,
    x_label: str = "Latency (s)",
    y_label: str = "QPS/Chip",
    x_tick_labels: list[str] | None = None,
    y_tick_labels: list[str] | None = None,
    x_log: bool = False,
    y_log: bool = False,
    x_lim: list[float] | None = None,
    y_lim: list[float] | None = None,
    y_lim_low: float | None = None,
    x_lim_low: float | None = None,
    y_num_ticks: int | None = None,
    markersize=5,
    label_font=12,
    title_font=12,
    legend_font=10,
    tick_font=10,
    legend_ncol=1,
    legend_loc=(0.0, 1.02),
    no_legend=False,
    figsize=(6, 4),
    plot_style="seaborn-v0_8-colorblind",
    text: str | None = None,
    text_font=10,
    text_loc=(0.5, 0.5),
    save_path: str | None = None,
    show_plot: bool = True,  # whether to show the plot
) -> None:
    """Plots curve from dataframe, e.g., x: latency, y: QPS."""
    sns.set_theme(style="whitegrid")
    plt.style.use(plot_style)

    default_colors = []
    for _, color in enumerate(plt.rcParams["axes.prop_cycle"]):
        default_colors.append(color["color"])
    num_colors = len(default_colors)
    if marker_style_list is None:
        marker_style_list = ["o", "X", "P", "v", "^", ".", "<", "1", "s", "+", "D"]

    num_components = len(df_list)

    if not figsize:
        figsize = (6, 4)
    _, ax = plt.subplots(1, 1, figsize=figsize)

    rects = []
    for i in range(num_components):
        x = df_list[i][x_key].to_numpy()
        if x_tick_labels is not None:
            unique_x = np.sort(np.unique(x))
            assert len(unique_x) <= len(x_tick_labels)
            x = range(len(x))
        y = df_list[i][y_key].to_numpy()
        if color_list is not None:
            color = default_colors[color_list[i] % num_colors]
        else:
            color = default_colors[i % num_colors]
        if linestyle_list is not None:
            linestyle = linestyle_list[i]
        else:
            linestyle = "solid"
        if marker_style_list is not None:
            marker_style = marker_style_list[i % len(marker_style_list)]
        else:
            marker_style = "o"
        rect = ax.plot(
            x,
            y,
            label=legend_list[i],
            color=color,
            linestyle=linestyle,
            marker=marker_style,
            markersize=markersize,
        )
        rects.append(rect)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            num = rect.get_height()
            y_pos = rect.get_y() + rect.get_height()
            str_print = f"{num:.2e}"
            ax.annotate(
                str_print,
                xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                xytext=(0, 0),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    if not no_legend:
        ax.legend(
            loc=legend_loc,
            ncol=legend_ncol,
            facecolor="white",
            framealpha=1,
            frameon=False,
            fontsize=legend_font,
        )
    if mark_number:
        autolabel(rects)

    ax.tick_params(
        top=False,
        bottom=True,
        left=True,
        right=False,
        labelleft=True,
        labelbottom=True,
        labelsize=tick_font,
    )

    ax.set_xlabel(x_label, fontsize=label_font)
    ax.set_ylabel(y_label, fontsize=label_font)
    if x_tick_labels is not None:
        ax.set_xticks(range(len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels, fontsize=tick_font)
    if y_tick_labels is not None:
        ax.set_yticklabels(y_tick_labels, fontsize=tick_font)
    if y_log:
        ax.set_yscale("log")
    if x_log:
        ax.set_xscale("log")
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if y_num_ticks:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=y_num_ticks))
    ax.set_title(f"{plot_name}", fontsize=title_font)

    if text is not None:
        plt.text(
            text_loc[0],
            text_loc[1],
            text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=text_font,
        )

    if y_lim_low is not None:
        plt.gca().set_ylim(bottom=y_lim_low)
    if x_lim_low is not None:
        plt.gca().set_xlim(left=x_lim_low)

    if save_path:
        plt.savefig(save_path, transparent=False, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_scatter_df(
    df_list: list[pd.DataFrame],  # 1D list: a number of dataframes
    legend_list: list[str],  # len == num_curves
    color_list: list[int] | None = None,
    marker_style_list: list[str] | None = None,
    x_key: str = "latency_s",
    y_key: str = "qps_per_chip",
    plot_name: str = "",
    mark_number: bool = False,
    x_label: str = "Latency (s)",
    y_label: str = "QPS per chip",
    x_log: bool = False,
    y_log: bool = False,
    x_lim: list[float] | None = None,
    y_lim: list[float] | None = None,
    markersize=50,
    label_font=12,
    title_font=12,
    legend_font=10,
    tick_font=10,
    legend_ncol=1,
    legend_loc=(0.0, 1.02),
    figsize=(6, 4),
    plot_style="seaborn-v0_8-pastel",
    save_path: str | None = None,
    show_plot: bool = True,  # whether to show the plot
) -> None:
    """Plots scatter curves from dataframes, e.g., x: latency, y: QPS."""
    sns.set_theme(style="whitegrid")
    plt.style.use(plot_style)

    # Generate default color cycle
    default_colors = [color["color"] for color in plt.rcParams["axes.prop_cycle"]]
    num_colors = len(default_colors)

    num_components = len(df_list)

    if not figsize:
        figsize = (6, 4)
    _, ax = plt.subplots(1, 1, figsize=figsize)

    scatter_plots = []
    for i in range(num_components):
        # Extract x and y data
        x = df_list[i][x_key].to_numpy()
        y = df_list[i][y_key].to_numpy()

        # Determine color
        if color_list is not None:
            color = default_colors[color_list[i % num_colors]]
        else:
            color = default_colors[i % num_colors]

        # Determine marker style
        if marker_style_list is not None:
            marker_style = marker_style_list[i]
        else:
            marker_style = "o"  # Default marker

        # Create scatter plot
        scatter = ax.scatter(
            x,
            y,
            label=legend_list[i],
            color=color,
            marker=marker_style,
            s=markersize,  # Marker size
            edgecolors="w",  # Optional: white edge for better visibility
            linewidth=0.5,
        )
        scatter_plots.append(scatter)

    def autolabel_scatter(scatter_plot):
        """Attach a text label to each point in the scatter plot."""
        offsets = scatter_plot.get_offsets()
        for point in offsets:
            x, y = point
            str_print = f"{y:.2e}"
            ax.annotate(
                str_print,
                xy=(x, y),
                xytext=(5, 5),  # Offset text by 5 points
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=8,
            )

    ax.legend(
        loc=legend_loc,
        ncol=legend_ncol,
        facecolor="white",
        framealpha=1,
        frameon=False,
        fontsize=legend_font,
    )

    if mark_number:
        for scatter in scatter_plots:
            autolabel_scatter(scatter)

    ax.tick_params(
        top=False,
        bottom=True,
        left=True,
        right=False,
        labelleft=True,
        labelbottom=True,
        labelsize=tick_font,
    )

    ax.set_xlabel(x_label, fontsize=label_font)
    ax.set_ylabel(y_label, fontsize=label_font)
    if y_log:
        ax.set_yscale("log")
    if x_log:
        ax.set_xscale("log")
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_title(f"{plot_name}", fontsize=title_font)

    if save_path:
        plt.savefig(save_path, transparent=False, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_linear_heatmap(
    data: np.ndarray,  # 2-d array
    color_id: int = 0,
    x_tick_labels: list[Any] = None,  # this is data dim 1
    y_tick_labels: list[Any] = None,  # this is data dim 0
    x_label: str = "Latency (s)",
    y_label: str = "QPS per chip",
    data_label: str = "TTFT Latency Reduction",
    plot_name: str | None = None,
    label_font=12,
    title_font=14,
    tick_font=10,
    figsize=(6, 4),
    fmt=".2f",  # number format
    save_path: str | None = None,
    show_plot: bool = True,  # whether to show the plot
):
    sns.set_theme(style="white")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    assert len(x_tick_labels) == data.shape[1]
    assert len(y_tick_labels) == data.shape[0]

    # matplotlib color map objects: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap_dict = {
        0: "RdBu",
        1: "RdYlGn",
        2: "coolwarm",
        3: "bwr",
        4: "seismic",
    }
    cmap = cmap_dict[color_id]

    ax_heatmap = seaborn.heatmap(
        data, cmap=cmap, cbar_kws={"label": data_label}, annot=True, fmt=fmt
    )
    # set colorbar label size by hacking: https://stackoverflow.com/questions/48586738/seaborn-heatmap-colorbar-label-font-size
    ax_heatmap.figure.axes[-1].yaxis.label.set_size(label_font)

    if x_tick_labels:
        ax.set_xticklabels(x_tick_labels, fontsize=tick_font)
    if y_tick_labels:
        ax.set_yticklabels(y_tick_labels, fontsize=tick_font)
    plt.yticks(rotation=0)

    ax.tick_params(
        length=0,
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=True,
        labelbottom=True,
        labelsize=tick_font,
    )

    ax.set_xlabel(x_label, fontsize=label_font, labelpad=10)
    ax.set_ylabel(y_label, fontsize=label_font, labelpad=10)
    if plot_name:
        ax.set_title(plot_name, fontsize=title_font)  # y=5
    # plt.text(2, len(y_tick_labels) + 2, "Linear Heatmap", fontsize=16)
    if save_path:
        plt.savefig(save_path, transparent=False, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()


def get_lower_is_better(metric_key):
    if metric_key == "latency_s_ttft":
        return True
    elif metric_key == "latency_s_tpot":
        return True
    elif metric_key == "latency_s":
        return True
    elif metric_key == "qps_per_chip":
        return False
    elif metric_key == "qps":
        return False
    else:
        raise ValueError("Unknown metric key: %s" % metric_key)


def get_label(metric_key):
    if metric_key == "latency_s_ttft":
        return "Latency TTFT (s)"
    elif metric_key == "latency_s_tpot":
        return "Latency TPOT (s)"
    elif metric_key == "latency_s":
        return "Latency End-to-End (s)"
    elif metric_key == "qps_per_chip":
        return "QPS per chip"
    elif metric_key == "qps":
        return "QPS"
    else:
        raise ValueError("Unknown metric key: %s" % metric_key)


def get_workload_breakdown(df_sweep_results, rago_instance):
    best_qps_row = df_sweep_results.loc[df_sweep_results["qps_per_chip"].idxmax()]
    if len(best_qps_row.shape) > 1:
        best_qps_row = best_qps_row.iloc[0]

    stages = rago_instance.stages
    qps = best_qps_row["qps"]
    num_chips_per_stage = {}
    qps_per_stage = {}
    workload_per_stage = {}

    workload_total = 0
    for stage in stages:

        if stage == "retrieval":
            num_retrieval_servers = best_qps_row["num_retrieval_servers"]
            num_chips_per_stage[stage] = (
                num_retrieval_servers * rago_instance.num_chips_per_server
            )
            df_stage = rago_instance.performance_pareto_dict[stage][
                num_retrieval_servers
            ]
            qps_per_stage[stage] = df_stage.loc[df_stage["qps"].idxmax()]["qps"]
        elif stage == "encode":
            num_chips_per_stage[stage] = best_qps_row[f"num_chips_{stage}"]
            qps_per_stage[stage] = 1 / rago_instance.get_encode_latency_s(
                num_chips_encode=num_chips_per_stage[stage],
                context_batch_size=1, 
            )
        else:
            num_chips_per_stage[stage] = best_qps_row[f"num_chips_{stage}"]
            df_stage = rago_instance.performance_pareto_dict[stage][
                num_chips_per_stage[stage]
            ]
            qps_per_stage[stage] = df_stage.loc[df_stage["qps"].idxmax()]["qps"]

        workload_per_stage[stage] = (
            num_chips_per_stage[stage] * qps / qps_per_stage[stage]
        )
        workload_total += workload_per_stage[stage]

    workload_distribution = []
    for stage in stages:
        workload_distribution.append(workload_per_stage[stage] / workload_total)

    return workload_distribution
