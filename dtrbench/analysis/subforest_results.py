"""
Plotting and reporting functions for the subforest benchmark.

This module provides functions to analyze the results of the subforest benchmark runs and generate plots and statistics.
It works with results files containing one or multiple datasets.
If multiple datasets are present, the results will be aggregated across datasets for the plots and statistics.
"""

import ast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
from matplotlib.patches import Patch
from matplotlib.colors import PowerNorm
from scipy.stats import spearmanr, wilcoxon


def read_subforest_benchmark_result(
    results_path, REP_NAMES, SUBFOREST_SIZES, SEL_STRATEGIES
):
    """Read and process the subforest selection results from a CSV file.

    Args:
        results_path (str): Path to the CSV file containing the subforest selection results.
        REP_NAMES (list[str]): List of representation names to filter the results.
        SUBFOREST_SIZES (list[int]): List of subforest sizes to filter the results.
        SEL_STRATEGIES (list[str]): List of selection strategies to filter the results.
    Returns:
        dict: A dictionary containing processed dataframes for representations, baselines, full forest, and single decision tree.
    """
    subf_df = pd.read_csv(results_path)

    col_map = {
        "dataset": "Dataset",
        "seed": "Seed",
        "fold": "Fold",
        "representation": "Representation",
        "selection_strategy": "Selection Strategy",
        "full_forest_size": "Full Forest Size",
        "subforest_size": "Subforest Size",
        "acc": "Accuracy",
        "macro_f1": "Macro F1",
        "mcc": "MCC",
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
        "minority_class": "Minority Class",
        "minority_support": "Minority Support",
        "minority_precision": "Minority Precision",
        "minority_recall": "Minority Recall",
        "minority_f1": "Minority F1",
        "feature_importances": "Feature Importances",
        "silhouette_score": "Silhouette Score",
        "agreement_with_full_forest": "Agreement with Full Forest",
        "indices": "Subforest Indices",
    }
    results_df = subf_df.rename(columns=col_map)

    baselines = ["Random", "Top OOB ACC", "Top OOB MCC"]
    results_df["is_baseline"] = results_df["Selection Strategy"].isin(baselines)
    results_df["is_full_forest"] = results_df["Selection Strategy"].isnull()
    full_forest_size = results_df["Full Forest Size"][0]

    subforest_sizes = (
        results_df["Subforest Size"]
        .dropna()
        .loc[lambda s: ~s.isin([1, full_forest_size])]
        .unique()
        .tolist()
    )

    _rep = results_df[
        results_df["Representation"].isin(REP_NAMES)
        & results_df["Subforest Size"].isin(subforest_sizes)
        & results_df["Selection Strategy"].isin(SEL_STRATEGIES)
    ].copy()
    _bl = results_df[
        results_df["Representation"].isin(baselines)
        & results_df["Subforest Size"].isin(subforest_sizes)
    ].copy()
    _ff = results_df[results_df["Representation"] == "Full Forest"].copy()
    _dt = results_df[results_df["Representation"] == "Single DT"].copy()

    shared_values = {
        "subforest_sizes": subforest_sizes,
        "rep": _rep,
        "bl": _bl,
        "ff": _ff,
        "dt": _dt,
    }

    if SUBFOREST_SIZES is not None:
        allowed_sizes = set(SUBFOREST_SIZES)
        shared_values["rep"] = shared_values["rep"][
            shared_values["rep"]["Subforest Size"].isin(allowed_sizes)
        ]
        shared_values["bl"] = shared_values["bl"][
            shared_values["bl"]["Subforest Size"].isin(allowed_sizes)
        ]
        shared_values["subforest_sizes"] = sorted(
            shared_values["rep"]["Subforest Size"].unique().tolist()
        )

    return shared_values


def plot_rf_compression(
    shared_values, output_dir, REP_NAMES, SEL_STRATEGIES, show_recovery
):
    """Plot the Random Forest Compression results, showing the relationship between subforest size and mean MCC for different representations and selection strategies.

    Aggregation:
        Results are aggregated by subforest size. Representation and selection strategy
        curves can either be averaged across the other dimension, fully combined, or
        reduced to the best-performing configuration per representation. All reported
        MCC values are averaged across datasets and folds.

    Args:
        shared_values (dict): Dictionary containing processed dataframes for different categories.
        output_dir (str): Directory where the plots will be saved.
        REP_NAMES (list[str]): List of representation names to be considered.
        SEL_STRATEGIES (list[str]): List of selection strategies to be considered.
    """

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Dynamic colorblind-friendly palette
    colorblind_palette = sns.color_palette("colorblind")

    def _get_dynamic_colors(n):
        """Return n distinct colors from a colorblind-friendly palette."""
        if n <= len(colorblind_palette):
            return colorblind_palette[:n]

        colors = list(colorblind_palette)
        cmap = plt.get_cmap("viridis")
        extra_n = n - len(colors)

        if extra_n == 1:
            extra_positions = [0.5]
        else:
            extra_positions = np.linspace(0.1, 0.9, extra_n)

        colors.extend(cmap(x) for x in extra_positions)

        return colors

    # Assign colors once so that the same series has the same color across all plots
    all_series = (
        list(REP_NAMES)
        + list(SEL_STRATEGIES)
        + [
            "Representation-based Methods",
            "Random",
            "Top OOB ACC",
            "Top OOB MCC",
        ]
    )
    all_series = list(dict.fromkeys(all_series))

    colors = _get_dynamic_colors(len(all_series))
    series_colors = dict(zip(all_series, colors))

    def _plot_line_plot(df, aggregate_values):
        _strat_rename = {
            "agglomerative-performance": "agp",
            "agglomerative": "ag",
            "combination-genetic": "ge",
            "combination-greedy": "gr",
            "combination-simulated_annealing": "sa",
            "density": "de",
            "k-medoid-performance": "kmp",
            "k-medoid": "km",
        }

        # Compute mean MCC per subforest size for each series
        if aggregate_values == "both":
            # one averaged line
            rep_lines = {
                "Representation-based Methods": df["rep"]
                .groupby("Subforest Size")["MCC"]
                .mean()
                .reset_index()
            }
        elif aggregate_values == "Representation":
            # average over representations -> one line per representation
            rep_lines = {}
            for rep in REP_NAMES:
                d = df["rep"][df["rep"]["Representation"] == rep]
                if not d.empty:
                    rep_lines[rep] = (
                        d.groupby("Subforest Size")["MCC"].mean().reset_index()
                    )
        elif aggregate_values == "Selection Strategy":
            # average over selection strategies -> one line per selection strategy
            rep_lines = {}
            for strategy in SEL_STRATEGIES:
                d = df["rep"][df["rep"]["Selection Strategy"] == strategy]
                if not d.empty:
                    rep_lines[strategy] = (
                        d.groupby("Subforest Size")["MCC"].mean().reset_index()
                    )
        elif aggregate_values == "Configuration":
            # average over configurations -> one line per representation as the best config for every representation is plotted
            rep_lines = {}
            for rep in REP_NAMES:
                d = df["rep"][df["rep"]["Representation"] == rep]
                if d.empty:
                    continue
                best_strategy = d.groupby("Selection Strategy")["MCC"].mean().idxmax()
                best = d[d["Selection Strategy"] == best_strategy]
                strategy_label = _strat_rename.get(best_strategy, best_strategy)
                rep_lines[f"{rep} ({strategy_label})"] = (
                    best.groupby("Subforest Size")["MCC"].mean().reset_index()
                )
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregate_values}")

        _bl_lines = {}
        for _rep in ["Random", "Top OOB ACC", "Top OOB MCC"]:
            _sub = df["bl"][df["bl"]["Representation"] == _rep]
            if not _sub.empty:
                _bl_lines[_rep] = (
                    _sub.groupby("Subforest Size")["MCC"].mean().reset_index()
                )

        # Full forest & Single DT: single horizontal value
        _ff_mcc = df["ff"]["MCC"].mean()
        _dt_mcc = df["dt"]["MCC"].mean()
        if show_recovery:
            ff_label = "Full Forest (Recovery = 1.000)"
            dt_label = f"Single DT (Recovery = {_dt_mcc:.3f})"
        else:
            ff_label = f"Full Forest (MCC = {_ff_mcc:.3f})"
            dt_label = f"Single DT (MCC = {_dt_mcc:.3f})"

        # Get color for a series
        def _get_series_color(label):
            if label in series_colors:
                return series_colors[label]

            # Configuration labels use the representation color
            if " (" in label:
                representation = label.split(" (", 1)[0]
                if representation in series_colors:
                    return series_colors[representation]

            # Fallback for unexpected series
            new_color = _get_dynamic_colors(len(series_colors) + 1)[-1]
            series_colors[label] = new_color
            return new_color

        fig_compression, ax_c = plt.subplots(figsize=(7, 4.5))

        # Advanced — shaded CI band
        # _rep_ci = df['rep'].groupby('Subforest Size')['MCC'].sem().reset_index().rename(columns={'MCC': 'SE'})
        # _rep_line = _rep_line.merge(_rep_ci, on='Subforest Size')

        for label, line in rep_lines.items():
            ax_c.plot(
                line["Subforest Size"],
                line["MCC"],
                marker="o",
                linewidth=2,
                color=_get_series_color(label),
                label=label,
            )

        # Baselines
        _bl_styles = {"Random": "--", "Top OOB ACC": "-.", "Top OOB MCC": ":"}
        for _rep, _ls in _bl_styles.items():
            if _rep not in _bl_lines:
                continue

            _d = _bl_lines[_rep]
            ax_c.plot(
                _d["Subforest Size"],
                _d["MCC"],
                linestyle=_ls,
                marker="s",
                linewidth=1.7,
                color=_get_series_color(_rep),
                label=_rep,
            )

        # Full forest & Single DT horizontal lines
        ax_c.axhline(
            _ff_mcc,
            color="black",
            linewidth=1.4,
            linestyle="--",
            label=ff_label,
        )
        ax_c.axhline(
            _dt_mcc,
            color="black",
            linewidth=1.4,
            linestyle=":",
            label=dt_label,
        )

        ax_c.set_xlabel("Subforest Size $k$")
        ax_c.set_xscale("log")
        ax_c.set_ylabel(
            "Recovery from Full Forest MCC" if show_recovery else "Mean MCC"
        )
        ax_c.set_xticks(df["subforest_sizes"])
        ax_c.set_xticklabels([str(k) for k in df["subforest_sizes"]])
        ax_c.legend(fontsize=9.5, frameon=True, loc="lower right")
        ax_c.set_title("RF Compression: MCC vs Subforest Size")
        sns.despine(ax=ax_c, top=True, right=True)

        suffix = "_recovery" if show_recovery else ""
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/fig_compression_{aggregate_values}{suffix}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig_compression)

    shared_values["bl"] = _average_random_baseline(shared_values["bl"])

    if show_recovery:
        shared_values = _normalize_to_full_forest(shared_values, "MCC")

    _plot_line_plot(shared_values, aggregate_values="both")
    _plot_line_plot(shared_values, aggregate_values="Selection Strategy")
    _plot_line_plot(shared_values, aggregate_values="Representation")
    _plot_line_plot(shared_values, aggregate_values="Configuration")

    print("Subforest Selection: Random Forest Compression - done.")
    print()


def plot_mcc_boxplots(
    shared_values, output_dir, REP_NAMES, SEL_STRATEGIES, show_recovery
):
    """Plot boxplots of MCC distributions for different representations and selection strategies, including baselines and full forest.

    Aggregation:
        MCC is averaged over folds for each dataset and subforest size. Boxplots
        display the resulting distributions grouped by representation or selection
        strategy and each dot represents a subforest of size k in some dataset.

    Args:
        shared_values (dict): Dictionary containing processed dataframes for different categories.
        output_dir (str): Directory where the plots will be saved.
        REP_NAMES (list[str]): List of representation names to be considered.
        SEL_STRATEGIES (list[str]): List of selection strategies to be considered.
    """

    def get_order(values, preferred_order):
        values = list(pd.unique(values))
        return [v for v in preferred_order if v in values] + [
            v for v in values if v not in preferred_order
        ]

    def _plot_boxplot(
        df, x, full_forest_mcc, single_DT_mcc, title, file_name, preferred_order=None
    ):
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        palette = sns.color_palette("colorblind")

        fig, ax = plt.subplots(figsize=(8, 4.5))

        order = None
        if preferred_order is not None:
            order = get_order(df[x], preferred_order)

        sns.boxplot(
            data=df,
            x=x,
            y="fold_mean",
            palette=palette,
            width=0.45,
            linewidth=1.2,
            showfliers=False,
            saturation=1,
            order=order,
            ax=ax,
            zorder=1,
            medianprops=dict(color="black", linewidth=2),
        )

        # Advanced — show individual points
        '''
        sns.stripplot(
            data=df,
            x=x,
            y="fold_mean",
            color="black",
            size=2.8,
            alpha=0.45,
            jitter=0.12,
            order=order,
            ax=ax,
            zorder=2,
        )
        '''

        for patch in ax.patches:
            patch.set_alpha(0.85)

        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", visible=False)

        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

        ax.axhline(
            full_forest_mcc,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Full Forest",
        )

        ax.axhline(
            single_DT_mcc,
            color="black",
            linestyle=":",
            linewidth=1.4,
            label="Single DT",
        )

        q_low = df["fold_mean"].quantile(0.02)
        q_high = df["fold_mean"].quantile(0.98)
        ymin = min(q_low, full_forest_mcc, single_DT_mcc) - 0.02
        ymax = max(q_high, full_forest_mcc, single_DT_mcc) + 0.02
        ax.set_ylim(ymin, ymax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Recovery of Full Forest MCC" if show_recovery else "Mean MCC")
        ax.legend(loc="lower right", frameon=False, fontsize=10)
        sns.despine(ax=ax)
        suffix = "_recovery" if show_recovery else ""
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_mcc_{file_name}{suffix}.png", dpi=600)

    shared_values["bl"] = _average_random_baseline(shared_values["bl"])
    if show_recovery:
        shared_values = _normalize_to_full_forest(shared_values, "MCC")

    rep_df = (
        shared_values["rep"]
        .groupby(["Representation", "Selection Strategy", "Dataset", "Subforest Size"])[
            "MCC"
        ]
        .agg(fold_mean="mean", fold_std="std")
        .reset_index()
    )
    bl_df = (
        shared_values["bl"]
        .groupby(["Representation", "Dataset", "Subforest Size"])["MCC"]
        .agg(fold_mean="mean", fold_std="std")
        .reset_index()
    )
    bl_df["Selection Strategy"] = bl_df["Representation"]
    agg_df = pd.concat([rep_df, bl_df], ignore_index=True)
    full_forest_mcc = shared_values["ff"]["MCC"].mean()
    single_DT_mcc = shared_values["dt"]["MCC"].mean()

    _plot_boxplot(
        agg_df,
        x="Representation",
        full_forest_mcc=full_forest_mcc,
        single_DT_mcc=single_DT_mcc,
        title="MCC Recovery Distribution by Representation"
        if show_recovery
        else "MCC Distribution by Representation",
        file_name="representation",
        preferred_order=REP_NAMES + ["Random", "Top OOB ACC", "Top OOB MCC"],
    )

    _plot_boxplot(
        agg_df,
        x="Selection Strategy",
        full_forest_mcc=full_forest_mcc,
        single_DT_mcc=single_DT_mcc,
        title="MCC Recovery Distribution by Selection Strategy"
        if show_recovery
        else "MCC Distribution by Selection Strategy",
        file_name="selection_strategy",
        preferred_order=SEL_STRATEGIES + ["Random", "Top OOB ACC", "Top OOB MCC"],
    )

    print("Subforest Selection: MCC Distribution - done.")
    print()


def plot_mcc_representation_selection_strategy(
    shared_values, output_dir, REP_NAMES, SEL_STRATEGIES, show_recovery
):
    """Plot a heatmap showing mean MCC for representation × selection strategy,
    with baseline methods displayed as columns on the right.

    Aggregation:
        Representation × selection strategy:
            averaged across folds, datasets, and subforest sizes.

        Baselines:
            averaged across datasets, then across the corresponding baseline
            entries, and displayed as separate columns.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    shared_values["bl"] = _average_random_baseline(shared_values["bl"])

    if show_recovery:
        shared_values = _normalize_to_full_forest(shared_values, "MCC")

    # Representation × selection strategy
    _heatmap_base = (
        shared_values["rep"]
        .groupby(
            [
                "Representation",
                "Selection Strategy",
                "Dataset",
                "Subforest Size",
            ]
        )["MCC"]
        .agg(fold_mean="mean", fold_std="std")
        .reset_index()
    )

    _heatmap_agg = (
        _heatmap_base
        .groupby(["Representation", "Selection Strategy"])
        .agg(
            final_mean=("fold_mean", "mean"),
            final_std=("fold_std", "mean"),
        )
        .reset_index()
    )

    # Baselines
    _fixed = pd.concat(
        [
            shared_values["bl"],
            shared_values["ff"],
            shared_values["dt"],
        ],
        ignore_index=True,
    )

    _fixed = (
        _fixed
        .groupby(["Representation", "Dataset"])["MCC"]
        .mean()
        .reset_index()
    )

    _baseline_values = (
        _fixed
        .groupby("Representation")["MCC"]
        .mean()
    )

    baseline_order = [
        "Full Forest",
        "Top OOB MCC",
        "Top OOB ACC",
        "Random",
        "Single DT",
    ]

    baseline_values = _baseline_values.reindex(
        [b for b in baseline_order if b in _baseline_values.index]
    )

    # Selection strategy names
    _strat_rename = {
        "agglomerative-performance": "agp",
        "agglomerative": "ag",
        "combination-genetic": "ge",
        "combination-greedy": "gr",
        "combination-simulated_annealing": "sa",
        "density": "de",
        "k-medoid-performance": "kmp",
        "k-medoid": "km",
    }

    _pivot_mcc = _heatmap_agg.pivot(
        index="Representation",
        columns="Selection Strategy",
        values="final_mean",
    )

    _pivot_mcc.columns = [
        _strat_rename.get(c, c)
        for c in _pivot_mcc.columns
    ]

    strategy_order = [
        _strat_rename.get(s, s)
        for s in SEL_STRATEGIES
    ]

    _pivot_mcc = _pivot_mcc.reindex(columns=strategy_order)

    # Representation names
    def _short_representation_name(r):
        return (
            r.replace("Tree Descriptor", "TD")
            .replace("Leaf Profile", "LP")
            .replace("Feature Graph", "FG")
            .replace("Topological Forest", "TF")
            .replace("INDTree", "ID")
        )

    _pivot_mcc.index = [
        _short_representation_name(r)
        for r in _pivot_mcc.index
    ]

    representation_order = [
        _short_representation_name(r)
        for r in REP_NAMES
    ]

    _pivot_mcc = _pivot_mcc.reindex(
        [r for r in representation_order if r in _pivot_mcc.index]
    )

    # Plot
    n_strategy_cols = len(_pivot_mcc.columns)
    n_baseline_cols = len(baseline_values)
    n_rows = len(_pivot_mcc.index)

    fig_heatmap_mcc, ax_hm = plt.subplots(
        figsize=(6, 3)
    )

    all_values = pd.concat(
        [
            _pivot_mcc.stack(),
            baseline_values,
        ]
    ).dropna()

    vmin = all_values.min()
    vmax = all_values.max()

    norm = PowerNorm(
        gamma=1.5,
        vmin=vmin,
        vmax=vmax,
    )

    cmap = plt.get_cmap("viridis")

    sns.heatmap(
        _pivot_mcc,
        ax=ax_hm,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 6},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={},
    )
    cbar = ax_hm.collections[0].colorbar
    cbar.set_label(
        "MCC Recovery of Full Forest"
        if show_recovery
        else "Mean MCC",
        fontsize=7,
    )
    cbar.ax.tick_params(labelsize=7)

    # Baseline blocks
    for i, (baseline_name, baseline_value) in enumerate(
        baseline_values.items()
    ):
        x = n_strategy_cols + i

        cell_color = cmap(norm(baseline_value))

        rect = plt.Rectangle(
            (x, 0),
            1,
            n_rows,
            facecolor=cell_color,
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )

        ax_hm.add_patch(rect)

        r, g, b, _ = cell_color

        luminance = (
            0.2126 * r
            + 0.7152 * g
            + 0.0722 * b
        )

        text_color = "white" if luminance < 0.5 else "black"

        ax_hm.text(
            x + 0.5,
            n_rows / 2,
            f"{baseline_value:.3f}",
            ha="center",
            va="center",
            fontsize=6,
            color=text_color,
            zorder=3,
        )

    ax_hm.set_xlim(
        0,
        n_strategy_cols + n_baseline_cols,
    )

    ax_hm.set_xticks(
        [
            *[
                i + 0.5
                for i in range(n_strategy_cols)
            ],
            *[
                n_strategy_cols + i + 0.5
                for i in range(n_baseline_cols)
            ],
        ]
    )

    ax_hm.set_xticklabels(
        [
            *_pivot_mcc.columns.tolist(),
            *baseline_values.index.tolist(),
        ]
    )

    # Separator
    ax_hm.axvline(
        n_strategy_cols,
        color="black",
        linewidth=1.5,
        zorder=10,
    )

    ax_hm.set_title(
        "Representation × Selection Strategy — "
        + ("Recovery" if show_recovery else "Mean MCC")
    )

    ax_hm.set_ylabel("Representation", fontsize=10)
    ax_hm.set_xlabel("")

    ax_hm.tick_params(
        axis="x",
        rotation=35,
        labelsize=7
    )

    ax_hm.tick_params(
        axis="y",
        rotation=0,
        labelsize=7
    )

    # Group labels
    n_total_cols = n_strategy_cols + n_baseline_cols

    strategy_center = (
        n_strategy_cols / 2
    ) / n_total_cols

    baseline_center = (
        n_strategy_cols + n_baseline_cols / 2
    ) / n_total_cols

    ax_hm.text(
        strategy_center,
        -0.5,
        "Selection Strategies",
        transform=ax_hm.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    ax_hm.text(
        baseline_center,
        -0.5,
        "Baselines",
        transform=ax_hm.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    suffix = "_recovery" if show_recovery else ""

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/heatmap_mcc{suffix}.png",
        dpi=600,
        bbox_inches="tight",
    )

    print(
        "Subforest Selection: Heatmap of mean MCC "
        "(Representation × Selection Strategy + Baselines) - done."
    )
    print()


def plot_std_representation_selection_strategy(
    shared_values, output_dir, REP_NAMES, SEL_STRATEGIES, show_recovery
):
    """Plot a heatmap showing the standard deviation of MCC for each combination of representation and selection strategy.

    Aggregation:
        Fold-wise MCC standard deviations are averaged over subforest sizes and, for
        recovery plots, normalized to the Full Forest and averaged across datasets.

    Args:
        shared_values (dict): Dictionary containing processed dataframes for different categories.
        output_dir (str): Directory where the plots will be saved.
        REP_NAMES (list[str]): List of representation names to be considered.
        SEL_STRATEGIES (list[str]): List of selection strategies to be considered.
    """

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    shared_values["bl"] = _average_random_baseline(shared_values["bl"])

    _heatmap_base = (
        shared_values["rep"]
        .groupby(
            [
                "Representation",
                "Selection Strategy",
                "Dataset",
                "Subforest Size",
            ]
        )["MCC"]
        .agg(fold_mean="mean", fold_std="std")
        .reset_index()
    )

    _heatmap_agg = (
        _heatmap_base
        .groupby(["Representation", "Selection Strategy", "Dataset"])
        .agg(
            final_mean=("fold_mean", "mean"),
            final_std=("fold_std", "mean"),
        )
        .reset_index()
    )

    _fixed = pd.concat(
        [
            shared_values["bl"],
            shared_values["ff"],
            shared_values["dt"],
        ],
        ignore_index=True,
    )

    _fixed = (
        _fixed
        .groupby(
            [
                "Representation",
                "Dataset",
                "Fold",
                "Subforest Size",
            ]
        )["MCC"]
        .mean()
        .reset_index()
    )

    _fixed = (
        _fixed
        .groupby(["Representation", "Dataset"])["MCC"]
        .std()
        .reset_index(name="fold_std")
    )

    _fixed = (
        _fixed
        .groupby(["Representation", "Dataset"])["fold_std"]
        .mean()
        .reset_index(name="final_std")
    )

    strategy_df = pd.DataFrame(
        {"Selection Strategy": SEL_STRATEGIES}
    )

    _fixed_expanded = (
        _fixed.assign(key=1)
        .merge(
            strategy_df.assign(key=1),
            on="key",
        )
        .drop(columns="key")
    )

    heatmap_df = pd.concat(
        [
            _heatmap_agg[
                [
                    "Representation",
                    "Selection Strategy",
                    "Dataset",
                    "final_std",
                ]
            ],
            _fixed_expanded[
                [
                    "Representation",
                    "Selection Strategy",
                    "Dataset",
                    "final_std",
                ]
            ],
        ],
        ignore_index=True,
    )

    if show_recovery:
        full_forest_std = heatmap_df[
            heatmap_df["Representation"] == "Full Forest"
        ][
            ["Dataset", "final_std"]
        ].rename(
            columns={"final_std": "full_forest_std"}
        )

        heatmap_df = heatmap_df.merge(
            full_forest_std,
            on="Dataset",
            how="left",
        )

        heatmap_df["final_std"] = (
            heatmap_df["final_std"]
            / heatmap_df["full_forest_std"]
        )

        heatmap_df = heatmap_df.drop(
            columns="full_forest_std"
        )

    heatmap_df = (
        heatmap_df
        .groupby(
            [
                "Representation",
                "Selection Strategy",
            ]
        )["final_std"]
        .mean()
        .reset_index()
    )

    _strat_rename = {
        "agglomerative-performance": "agp",
        "agglomerative": "ag",
        "combination-genetic": "ge",
        "combination-greedy": "gr",
        "combination-simulated_annealing": "sa",
        "density": "de",
        "k-medoid-performance": "kmp",
        "k-medoid": "km",
    }

    _pivot_std = heatmap_df.pivot(
        index="Representation",
        columns="Selection Strategy",
        values="final_std",
    )

    _pivot_std.columns = [
        _strat_rename.get(c, c)
        for c in _pivot_std.columns
    ]

    strategy_order = [
        _strat_rename.get(s, s)
        for s in SEL_STRATEGIES
    ]

    _pivot_std = _pivot_std.reindex(
        columns=strategy_order
    )

    _pivot_std.index = [
        r.replace("Leaf Profile", "LP")
        .replace("Tree Descriptor", "TD")
        .replace("Feature Graph", "FG")
        .replace("Topological Forest", "TF")
        .replace("INDTree", "ID")
        for r in _pivot_std.index
    ]

    representation_order = [
        r.replace("Tree Descriptor", "TD")
        .replace("Leaf Profile", "LP")
        .replace("Feature Graph", "FG")
        .replace("Topological Forest", "TF")
        .replace("INDTree", "ID")
        for r in REP_NAMES
    ]

    _pivot_std = _pivot_std.reindex(
        [
            r
            for r in representation_order
            if r in _pivot_std.index
        ]
    )

    baseline_rows = [
        "Full Forest",
        "Top OOB MCC",
        "Top OOB ACC",
        "Random",
        "Single DT",
    ]

    baseline_values = heatmap_df[
        heatmap_df["Representation"].isin(baseline_rows)
    ].groupby("Representation")["final_std"].mean()

    baseline_values = baseline_values.reindex(
        [
            b
            for b in baseline_rows
            if b in baseline_values.index
        ]
    )

    fig_heatmap_std, ax_hs = plt.subplots(
        figsize=(6, 3)
    )

    all_values = pd.concat(
        [
            _pivot_std.stack(),
            baseline_values,
        ]
    ).dropna()

    vmin = all_values.min()
    vmax = all_values.max()

    norm = plt.Normalize(
        vmin=vmin,
        vmax=vmax,
    )

    cmap = plt.get_cmap("OrRd")

    sns.heatmap(
        _pivot_std,
        ax=ax_hs,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 6},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={},
    )
    cbar = ax_hs.collections[0].colorbar
    cbar.set_label(
        "Std ratio to Full Forest"
        if show_recovery
        else "Std of MCC",
        fontsize=7,
    )
    cbar.ax.tick_params(labelsize=7)

    n_strategy_cols = len(_pivot_std.columns)
    n_baseline_cols = len(baseline_values)
    n_rows = len(_pivot_std.index)

    for i, (baseline_name, baseline_value) in enumerate(
        baseline_values.items()
    ):
        x = n_strategy_cols + i

        cell_color = cmap(norm(baseline_value))

        rect = plt.Rectangle(
            (x, 0),
            1,
            n_rows,
            facecolor=cell_color,
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )

        ax_hs.add_patch(rect)

        r, g, b, _ = cell_color

        luminance = (
            0.2126 * r
            + 0.7152 * g
            + 0.0722 * b
        )

        text_color = (
            "white"
            if luminance < 0.5
            else "black"
        )

        ax_hs.text(
            x + 0.5,
            n_rows / 2,
            f"{baseline_value:.3f}",
            ha="center",
            va="center",
            fontsize=6,
            color=text_color,
            zorder=3,
        )

    ax_hs.set_xlim(
        0,
        n_strategy_cols + n_baseline_cols,
    )

    ax_hs.set_xticks(
        [
            *[
                i + 0.5
                for i in range(n_strategy_cols)
            ],
            *[
                n_strategy_cols + i + 0.5
                for i in range(n_baseline_cols)
            ],
        ]
    )

    ax_hs.set_xticklabels(
        [
            *_pivot_std.columns.tolist(),
            *baseline_values.index.tolist(),
        ]
    )

    ax_hs.axvline(
        n_strategy_cols,
        color="black",
        linewidth=1.5,
        zorder=10,
    )

    ax_hs.set_title(
        "Representation × Selection Strategy — "
        + (
            "Recovery Standard Deviation"
            if show_recovery
            else "MCC Standard Deviation"
        )
    )

    ax_hs.set_ylabel("Representation", fontsize=10)
    ax_hs.set_xlabel("")

    ax_hs.tick_params(
        axis="x",
        rotation=35,
        labelsize=7,
    )

    ax_hs.tick_params(
        axis="y",
        rotation=0,
        labelsize=7,
    )

    n_total_cols = (
        n_strategy_cols + n_baseline_cols
    )

    strategy_center = (
        n_strategy_cols / 2
    ) / n_total_cols

    baseline_center = (
        n_strategy_cols
        + n_baseline_cols / 2
    ) / n_total_cols

    ax_hs.text(
        strategy_center,
        -0.5,
        "Selection Strategies",
        transform=ax_hs.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    ax_hs.text(
        baseline_center,
        -0.5,
        "Baselines",
        transform=ax_hs.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    
    )

    suffix = "_recovery" if show_recovery else ""

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/heatmap_std{suffix}.png",
        dpi=600,
        bbox_inches="tight",
    )

    print(
        "Subforest Selection: Heatmap of MCC standard deviation "
        "(Representation × Selection Strategy + Baselines) - done."
    )
    print()


def plot_kendalls_w_vs_config(
    shared_values, output_dir, REP_NAMES, SEL_STRATEGIES, show_recovery
):
    """Plot Kendall's W for feature importance rankings across different configurations of representations and selection strategies.

    Aggregation:
        Kendall's W is computed across folds and averaged across datasets and
        subforest sizes for each representation–selection strategy combination.

    Args:
        shared_values (dict): Dictionary containing processed dataframes for different categories.
        output_dir (str): Directory where the plots will be saved.
        REP_NAMES (list[str]): List of representation names to be considered.
        SEL_STRATEGIES (list[str]): List of selection strategies to be considered.
    """

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("colorblind")

    shared_values["bl"] = _average_random_baseline(shared_values["bl"])

    def _safe_parse_fi(x):
        if pd.isnull(x) or str(x).strip() in ("", "nan", "None"):
            return None
        try:
            return np.array(ast.literal_eval(str(x)), dtype=float)
        except Exception:
            return None

    def _kendall_w(fi_list):
        valid = [f for f in fi_list if f is not None]
        if len(valid) < 2:
            return np.nan
        min_len = min(len(v) for v in valid)
        valid = [v[:min_len] for v in valid]
        rm = np.array([pd.Series(v).rank(ascending=False).values for v in valid])
        m, n = rm.shape
        if m < 2 or n < 2:
            return np.nan
        col_sums = rm.sum(axis=0)
        S = np.sum((col_sums - col_sums.mean()) ** 2)
        return 12 * S / (m**2 * (n**3 - n))

    shared_values["rep"]["FI_arr"] = shared_values["rep"]["Feature Importances"].apply(
        _safe_parse_fi
    )
    shared_values["bl"]["FI_arr"] = shared_values["bl"]["Feature Importances"].apply(
        _safe_parse_fi
    )

    _strat_rename = {
        "agglomerative-performance": "agp",
        "agglomerative": "ag",
        "combination-genetic": "ge",
        "combination-greedy": "gr",
        "combination-simulated_annealing": "sa",
        "density": "de",
        "k-medoid-performance": "kmp",
        "k-medoid": "km",
    }

    _kw_records = []

    # representations
    for (ds, rep, strat, k), grp in shared_values["rep"].groupby(
        ["Dataset", "Representation", "Selection Strategy", "Subforest Size"]
    ):
        _kw = _kendall_w(grp["FI_arr"].tolist())
        _kw_records.append(
            {
                "Representation": rep,
                "Selection Strategy": strat,
                "Subforest Size": k,
                "Kendall_W": _kw,
            }
        )

    # baselines
    for (ds, rep, k), grp in shared_values["bl"].groupby(
        ["Dataset", "Representation", "Subforest Size"]
    ):
        _kw = _kendall_w(grp["FI_arr"].tolist())
        for strat in SEL_STRATEGIES:
            _kw_records.append(
                {
                    "Representation": rep,
                    "Selection Strategy": "",
                    "Subforest Size": k,
                    "Kendall_W": _kw,
                }
            )

    _kw_df = pd.DataFrame(_kw_records)

    _kw_cfg = (
        _kw_df.groupby(["Representation", "Selection Strategy"])["Kendall_W"]
        .mean()
        .reset_index()
    )

    _kw_cfg["Config"] = _kw_cfg.apply(
        lambda r: (
            {
                "Tree Descriptor": "TD",
                "Leaf Profile": "LP",
                "Feature Graph": "FG",
                "Topological Forest": "TF",
                "INDTree": "ID",
            }.get(r["Representation"], r["Representation"])
            if r["Selection Strategy"] == ""
            else {
                "Tree Descriptor": "TD",
                "Leaf Profile": "LP",
                "Feature Graph": "FG",
                "Topological Forest": "TF",
                "INDTree": "ID",
            }.get(r["Representation"], r["Representation"])
            + "+"
            + _strat_rename.get(
                r["Selection Strategy"],
                r["Selection Strategy"],
            )
        ),
        axis=1,
    )

    _kw_cfg = _kw_cfg.sort_values(
        "Kendall_W",
        ascending=True,
    ).reset_index(drop=True)

    # barplot
    # Color bars by Representation
    _rep_color_map = {
        "Tree Descriptor": palette[0],
        "Leaf Profile": palette[1],
        "Feature Graph": palette[2],
        "Topological Forest": palette[3],
        "INDTree": palette[4],
        "Top OOB MCC": palette[6],
        "Top OOB ACC": palette[7],
        "Random": palette[8],
    }

    _bar_colors = [
        _rep_color_map[r]
        for r in _kw_cfg["Representation"]
    ]

    # Reference line for full forest & single DT mean Kendall W
    _ff_fi = shared_values["ff"].copy()
    _ff_fi["FI_arr"] = _ff_fi["Feature Importances"].apply(
        _safe_parse_fi
    )

    _ff_kw_vals = []

    for (ds, k), grp in _ff_fi.groupby(
        ["Dataset", "Subforest Size"]
    ):
        _ff_kw_vals.append(
            _kendall_w(grp["FI_arr"].tolist())
        )

    _ff_kw_mean = np.nanmean(_ff_kw_vals)

    _dt_fi = shared_values["dt"].copy()
    _dt_fi["FI_arr"] = _dt_fi["Feature Importances"].apply(
        _safe_parse_fi
    )

    _dt_kw_vals = []

    for (ds, k), grp in _dt_fi.groupby(
        ["Dataset", "Subforest Size"]
    ):
        _dt_kw_vals.append(
            _kendall_w(grp["FI_arr"].tolist())
        )

    _dt_kw_mean = np.nanmean(_dt_kw_vals)

    fig_kendall, ax_kw = plt.subplots(figsize=(10, 13))

    ax_kw.barh(
        range(len(_kw_cfg)),
        _kw_cfg["Kendall_W"],
        color=_bar_colors,
        edgecolor="white",
        linewidth=0.3,
    )

    ax_kw.set_yticks(range(len(_kw_cfg)))
    ax_kw.set_yticklabels(
        _kw_cfg["Config"],
        fontsize=9,
    )

    ax_kw.set_xlabel("Kendall's $W$")
    ax_kw.set_ylabel(
        "Configuration (Representation + Selection Strategy)"
    )

    ax_kw.set_title(
        "Feature Importance Stability (Kendall's $W$) per Configuration — averaged across datasets, folds, subforest sizes"
    )

    ax_kw.set_ylim(
        -0.5,
        len(_kw_cfg) - 0.5,
    )

    xmin = min(
        _kw_cfg["Kendall_W"].min(),
        _ff_kw_mean,
        _dt_kw_mean,
    )

    xmax = max(
        _kw_cfg["Kendall_W"].max(),
        _ff_kw_mean,
        _dt_kw_mean,
    )

    margin = 0.05 * (xmax - xmin)

    ax_kw.set_xlim(
        xmin - margin,
        xmax + margin,
    )

    # Reference line for full forest
    ax_kw.axvline(
        _ff_kw_mean,
        color="black",
        linewidth=1.4,
        linestyle="--",
        label=f"Full Forest $W$ = {_ff_kw_mean:.3f}",
    )

    # Reference line for single DT
    ax_kw.axvline(
        _dt_kw_mean,
        color="black",
        linewidth=1.4,
        linestyle=":",
        label=f"Single DT $W$ = {_dt_kw_mean:.3f}",
    )

    # Legend
    _legend_handles = [
        Patch(facecolor=_rep_color_map[r], label=r)
        for r in REP_NAMES
        + ["Random", "Top OOB ACC", "Top OOB MCC"]
    ]

    ax_kw.legend(
        handles=_legend_handles
        + [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.4,
                linestyle="--",
                label=f"Full Forest $W$ = {_ff_kw_mean:.3f}",
            ),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.4,
                linestyle=":",
                label=f"Single DT $W$ = {_dt_kw_mean:.3f}",
            ),
        ],
        fontsize=11,
        frameon=True,
        loc="lower right",
        ncol=2,
    )

    sns.despine(
        ax=ax_kw,
        top=True,
        right=True,
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/fig_kendall_configurations_barplot.png",
        dpi=600,
    )

    # heatmap
    _fixed_kw = []

    # Full Forest
    for (ds, k), grp in _ff_fi.groupby(
        ["Dataset", "Subforest Size"]
    ):
        _fixed_kw.append(
            {
                "Representation": "Full Forest",
                "Selection Strategy": "",
                "Kendall_W": _kendall_w(
                    grp["FI_arr"].tolist()
                ),
            }
        )

    # Single DT
    for (ds, k), grp in _dt_fi.groupby(
        ["Dataset", "Subforest Size"]
    ):
        _fixed_kw.append(
            {
                "Representation": "Single DT",
                "Selection Strategy": "",
                "Kendall_W": _kendall_w(
                    grp["FI_arr"].tolist()
                ),
            }
        )

    # Random baseline
    for (ds, rep, k), grp in shared_values["bl"].groupby(
        ["Dataset", "Representation", "Subforest Size"]
    ):
        _fixed_kw.append(
            {
                "Representation": rep,
                "Selection Strategy": "",
                "Kendall_W": _kendall_w(
                    grp["FI_arr"].tolist()
                ),
            }
        )

    _fixed_kw = pd.DataFrame(_fixed_kw)

    # Average baseline values
    _fixed_kw = (
        _fixed_kw
        .groupby("Representation")["Kendall_W"]
        .mean()
        .reset_index()
    )

    # Expand baselines over strategies to create heatmap rows
    _fixed_expanded = (
        _fixed_kw.assign(key=1)
        .merge(
            pd.DataFrame(
                {
                    "Selection Strategy": SEL_STRATEGIES,
                    "key": 1,
                }
            ),
            on="key",
        )
        .drop(columns="key")
    )

    _heatmap_kw = pd.concat(
        [
            _kw_cfg[
                [
                    "Representation",
                    "Selection Strategy",
                    "Kendall_W",
                ]
            ],
            _fixed_expanded,
        ],
        ignore_index=True,
    )

    _pivot_kw = _heatmap_kw.pivot(
        index="Representation",
        columns="Selection Strategy",
        values="Kendall_W",
    )

    _pivot_kw.columns = [
        _strat_rename.get(c, c)
        if c != ""
        else c
        for c in _pivot_kw.columns
    ]

    strategy_order = [
        _strat_rename.get(s, s)
        for s in SEL_STRATEGIES
    ]

    _pivot_kw = _pivot_kw.reindex(
        columns=strategy_order
    )

    # Rename representations
    _pivot_kw.index = [
        r.replace("Tree Descriptor", "TD")
        .replace("Leaf Profile", "LP")
        .replace("Feature Graph", "FG")
        .replace("Topological Forest", "TF")
        .replace("INDTree", "ID")
        for r in _pivot_kw.index
    ]

    representation_order = [
        r.replace("Tree Descriptor", "TD")
        .replace("Leaf Profile", "LP")
        .replace("Feature Graph", "FG")
        .replace("Topological Forest", "TF")
        .replace("INDTree", "ID")
        for r in REP_NAMES
    ]

    final_order = (
        ["Full Forest"]
        + representation_order
        + ["Top OOB MCC", "Top OOB ACC", "Random", "Single DT"]
    )

    _pivot_kw = _pivot_kw.reindex(
        [
            r
            for r in final_order
            if r in _pivot_kw.index
        ]
    )

    # Baseline values
    baseline_rows = [
        "Full Forest",
        "Top OOB MCC",
        "Top OOB ACC",
        "Random",
        "Single DT",
    ]

    _pivot_kw = _pivot_kw.drop(
        index=baseline_rows,
        errors="ignore",
    )

    baseline_values = _fixed_kw.set_index(
        "Representation"
    )["Kendall_W"]

    baseline_values = baseline_values.reindex(
        [
            b
            for b in baseline_rows
            if b in baseline_values.index
        ]
    )

    fig_kw_heatmap, ax_kw_hm = plt.subplots(
        figsize=(6, 3)
    )

    all_values = pd.concat(
        [
            _pivot_kw.stack(),
            baseline_values,
        ]
    ).dropna()

    vmin = all_values.min()
    vmax = all_values.max()

    norm = plt.Normalize(
        vmin=vmin,
        vmax=vmax,
    )

    cmap = plt.get_cmap("viridis")

    sns.heatmap(
        _pivot_kw,
        ax=ax_kw_hm,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 6},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={},
    )
    cbar = ax_kw_hm.collections[0].colorbar
    cbar.set_label("Mean Kendall's $W$", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    n_strategy_cols = len(_pivot_kw.columns)
    n_baseline_cols = len(baseline_values)
    n_rows = len(_pivot_kw.index)

    for i, (baseline_name, baseline_value) in enumerate(
        baseline_values.items()
    ):
        x = n_strategy_cols + i

        cell_color = cmap(norm(baseline_value))

        rect = plt.Rectangle(
            (x, 0),
            1,
            n_rows,
            facecolor=cell_color,
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )

        ax_kw_hm.add_patch(rect)

        r, g, b, _ = cell_color

        luminance = (
            0.2126 * r
            + 0.7152 * g
            + 0.0722 * b
        )

        text_color = (
            "white"
            if luminance < 0.5
            else "black"
        )

        ax_kw_hm.text(
            x + 0.5,
            n_rows / 2,
            f"{baseline_value:.3f}",
            ha="center",
            va="center",
            fontsize=6,
            color=text_color,
            zorder=3,
        )

    ax_kw_hm.set_xlim(
        0,
        n_strategy_cols + n_baseline_cols,
    )

    ax_kw_hm.set_xticks(
        [
            *[
                i + 0.5
                for i in range(n_strategy_cols)
            ],
            *[
                n_strategy_cols + i + 0.5
                for i in range(n_baseline_cols)
            ],
        ]
    )

    ax_kw_hm.set_xticklabels(
        [
            *_pivot_kw.columns.tolist(),
            *baseline_values.index.tolist(),
        ]
    )

    ax_kw_hm.axvline(
        n_strategy_cols,
        color="black",
        linewidth=1.5,
        zorder=10,
    )

    ax_kw_hm.set_title(
        "Representation × Selection Strategy — "
        "Feature Importance Stability"
    )

    ax_kw_hm.set_ylabel("Representation", fontsize=10)
    ax_kw_hm.set_xlabel("")

    ax_kw_hm.tick_params(
        axis="x",
        rotation=35,
        labelsize=7,
    )

    ax_kw_hm.tick_params(
        axis="y",
        rotation=0,
        labelsize=7,
    )

    n_total_cols = (
        n_strategy_cols + n_baseline_cols
    )

    strategy_center = (
        n_strategy_cols / 2
    ) / n_total_cols

    baseline_center = (
        n_strategy_cols
        + n_baseline_cols / 2
    ) / n_total_cols

    ax_kw_hm.text(
        strategy_center,
        -0.5,
        "Selection Strategies",
        transform=ax_kw_hm.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    ax_kw_hm.text(
        baseline_center,
        -0.5,
        "Baselines",
        transform=ax_kw_hm.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/heatmap_kendall_configurations.png",
        dpi=600,
        bbox_inches="tight",
    )

    print(
        "Subforest Selection: Kendall's W vs Configuration - done."
    )
    print()


def plot_spearman_vs_subforest_size(
    shared_values, output_dir, REP_NAMES, SEL_STRATEGIES, show_recovery
):
    """Plot Spearman correlation of feature importance rankings between subforests and full forests across different representations and selection strategies.

    Aggregation:
        Spearman correlations with the corresponding full forest are averaged across
        datasets, seeds, and folds for each representation, selection strategy, and
        subforest size.

    Args:
        shared_values (dict): Dictionary containing processed dataframes for different categories.
        output_dir (str): Directory where the plots will be saved.
        REP_NAMES (list[str]): List of representation names to be considered.
        SEL_STRATEGIES (list[str]): List of selection strategies to be considered.
    """

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("colorblind")

    def _safe_parse_fi(x):
        if pd.isnull(x) or str(x).strip() in ("", "nan", "None"):
            return None
        try:
            return np.array(ast.literal_eval(str(x)), dtype=float)
        except Exception:
            return None

    shared_values["rep"]["FI_arr"] = shared_values["rep"]["Feature Importances"].apply(
        _safe_parse_fi
    )
    shared_values["bl"]["FI_arr"] = shared_values["bl"]["Feature Importances"].apply(
        _safe_parse_fi
    )

    _rep_color_map = {
        "Tree Descriptor": palette[0],
        "Leaf Profile": palette[1],
        "Feature Graph": palette[2],
        "Topological Forest": palette[3],
        "INDTree": palette[4],
        "Random": palette[6],
        "Top OOB ACC": palette[7],
        "Top OOB MCC": palette[8],
    }

    _ff = shared_values["ff"].copy()
    _ff["FI_arr"] = _ff["Feature Importances"].apply(_safe_parse_fi)
    _dt = shared_values["dt"].copy()
    _dt["FI_arr"] = _dt["Feature Importances"].apply(_safe_parse_fi)

    _sc_records = []
    for _, row in shared_values["rep"].iterrows():
        ff = _ff[
            (_ff["Dataset"] == row["Dataset"])
            & (_ff["Fold"] == row["Fold"])
            & (_ff["Seed"] == row["Seed"])
        ]
        if len(ff) != 1:
            continue
        fi_sub = row["FI_arr"]
        fi_full = ff.iloc[0]["FI_arr"]
        if fi_sub is None or fi_full is None:
            continue

        rho, sc_p = spearmanr(fi_sub, fi_full)

        _sc_records.append(
            {
                "Representation": row["Representation"],
                "Selection Strategy": row["Selection Strategy"],
                "Subforest Size": row["Subforest Size"],
                "Spearman": rho,
            }
        )

    for _, row in shared_values["bl"].iterrows():
        ff = _ff[
            (_ff["Dataset"] == row["Dataset"])
            & (_ff["Fold"] == row["Fold"])
            & (_ff["Seed"] == row["Seed"])
        ]
        if len(ff) != 1:
            continue
        fi_sub = row["FI_arr"]
        fi_full = ff.iloc[0]["FI_arr"]
        if fi_sub is None or fi_full is None:
            continue

        rho, sc_p = spearmanr(fi_sub, fi_full)

        for strat in SEL_STRATEGIES:
            _sc_records.append(
                {
                    "Representation": row["Representation"],
                    "Selection Strategy": strat,
                    "Subforest Size": row["Subforest Size"],
                    "Spearman": rho,
                }
            )

    _dt_records = []

    for _, row in _dt.iterrows():
        ff = _ff[
            (_ff["Dataset"] == row["Dataset"])
            & (_ff["Fold"] == row["Fold"])
            & (_ff["Seed"] == row["Seed"])
        ]
        if len(ff) != 1:
            continue
        fi_dt = row["FI_arr"]
        fi_ff = ff.iloc[0]["FI_arr"]
        if fi_dt is None or fi_ff is None:
            continue

        rho, _ = spearmanr(fi_dt, fi_ff)

        _dt_records.append(rho)

    _dt_mean = np.nanmean(_dt_records)

    _spear_df = pd.DataFrame(_sc_records)
    _spear_cfg = (
        _spear_df.groupby(["Representation", "Selection Strategy", "Subforest Size"])[
            "Spearman"
        ]
        .mean()
        .reset_index()
        .sort_values("Subforest Size")
    )

    _baseline_styles = {
        "Random": ("s", "--"),
        "Top OOB ACC": ("s", "-."),
        "Top OOB MCC": ("s", ":"),
    }
    _spear_cfg["Style"] = _spear_cfg["Representation"].apply(
        lambda x: x if x in _baseline_styles else "Representation"
    )

    color_to_rep = {
        matplotlib.colors.to_hex(color): rep for rep, color in _rep_color_map.items()
    }

    g = sns.relplot(
        data=_spear_cfg,
        x="Subforest Size",
        y="Spearman",
        hue="Representation",
        col="Selection Strategy",
        col_order=SEL_STRATEGIES,
        kind="line",
        marker="o",
        col_wrap=4,
        palette=_rep_color_map,
        height=3,
    )

    for ax in g.axes.flat:
        for line in ax.lines:
            color = matplotlib.colors.to_hex(line.get_color())
            if color in color_to_rep:
                rep = color_to_rep[color]
                if rep in _baseline_styles:
                    marker, linestyle = _baseline_styles[rep]
                else:
                    marker, linestyle = "o", "-"
                line.set_marker(marker)
                line.set_linestyle(linestyle)

    g.set_titles("{col_name}")
    g.legend.set_title("Representation /\n Baseline")
    g.set_axis_labels("Subforest Size", "Spearman correlation")
    g.set(ylim=(0, 1))
    for ax in g.axes.flat:
        ax.axhline(
            _dt_mean, color="black", linestyle=":", linewidth=1.4, label="Single DT"
        )

    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    legend_order = REP_NAMES + list(_baseline_styles.keys())
    label_to_handle = dict(zip(labels, handles))
    handles = [label_to_handle[lb] for lb in legend_order if lb in label_to_handle]
    labels = [lb for lb in legend_order if lb in label_to_handle]
    handles.append(
        plt.Line2D(
            [0], [0], color="black", linestyle=":", linewidth=1.4, label="Single DT"
        )
    )
    labels.append("Single DT")
    g.legend.remove()
    g.fig.legend(
        handles, labels, loc="center right", title="Representation /\nBaseline"
    )

    g.savefig(f"{output_dir}/fig_spearman_full_forest.png", dpi=600)
    plt.close(g.figure)

    print("Subforest Selection: Spearman's R vs Subforest Size - done.")
    print()


def print_representation_vs_subforest_size(
    shared_values, output_dir, REP_NAMES, show_recovery
):
    """Generate a LaTeX table summarizing the mean MCC and standard deviation for each representation and subforest size.

    Aggregation:
        Results are averaged across folds and datasets; representation-based methods
        are additionally averaged over selection strategies for each subforest size.

    Args:
        shared_values (dict): Dictionary containing processed dataframes for different categories.
        output_dir (str): Directory where the LaTeX table will be saved.
        REP_NAMES (list[str]): List of representation names to be considered.
    """
    shared_values["bl"] = _average_random_baseline(shared_values["bl"])
    if show_recovery:
        shared_values = _normalize_to_full_forest(shared_values, "MCC")

    _ff = shared_values["ff"].groupby(["Dataset"])["MCC"].mean()
    ff_mean = _ff.mean()
    ff_std = _ff.std()
    _dt = shared_values["dt"].groupby(["Dataset"])["MCC"].mean()
    dt_mean = _dt.mean()
    dt_std = _dt.std()
    _bl = (
        shared_values["bl"]
        .groupby(["Representation", "Dataset", "Subforest Size"])["MCC"]
        .agg(fold_mean="mean", fold_std="std")
        .reset_index()
    )
    _bl = (
        _bl.groupby(["Representation", "Subforest Size"])
        .agg(
            avg_mcc=("fold_mean", "mean"),
            avg_std=("fold_std", "mean"),
        )
        .reset_index()
    )

    _rep_strat_ds_k = (
        shared_values["rep"]
        .groupby(["Representation", "Selection Strategy", "Dataset", "Subforest Size"])[
            "MCC"
        ]
        .agg(fold_mean="mean", fold_std="std")
        .reset_index()
    )
    _rep_k = (
        _rep_strat_ds_k.groupby(["Representation", "Subforest Size"])
        .agg(avg_mcc=("fold_mean", "mean"), avg_std=("fold_std", "mean"))
        .reset_index()
    )
    _best_rep_per_k = (
        _rep_k.groupby("Subforest Size")
        .apply(lambda d: d.loc[d["avg_mcc"].idxmax(), "Representation"])
        .to_dict()
    )
    _rep_scores = _rep_k[["Representation", "Subforest Size", "avg_mcc"]].copy()
    _bl_scores = _bl[["Representation", "Subforest Size", "avg_mcc"]].copy()
    _all_scores = pd.concat([_rep_scores, _bl_scores], ignore_index=True)
    _best_overall_per_k = (
        _all_scores.groupby("Subforest Size")["avg_mcc"].max().to_dict()
    )

    def _fmt_cell(mcc, std, underline=False, bold=False):
        value = f"{mcc:.3f}"
        if underline:
            value = f"\\underline{{{value}}}"
        if bold:
            value = f"\\mathbf{{{value}}}"
        return f"${value} \\pm {std:.3f}$"

    metric_name = "Recovery" if show_recovery else "MCC"
    _col_header = " & ".join([f"$k={k}$" for k in shared_values["subforest_sizes"]])
    _latex_table1_lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\caption{Average test "
        + metric_name
        + " (standard deviation in parentheses) per representation and subforest size, averaged across all selection strategies, datasets, and folds. Underline: best representation for each $k$. Bold: best overall method for each $k$.}",
        "  \\label{tab:rep_mcc}",
        f"  \\begin{{tabular}}{{l{'c' * len(shared_values['subforest_sizes'])}}}",
        "    \\toprule",
        f"    \\textbf{{Representation}} & {_col_header} \\\\",
        "    \\midrule",
    ]

    _latex_table1_lines += [
        f"    Full Forest & \\multicolumn{{{len(shared_values['subforest_sizes'])}}}{{c}}{{{_fmt_cell(ff_mean, ff_std)}}} \\\\",
        f"    Single DT & \\multicolumn{{{len(shared_values['subforest_sizes'])}}}{{c}}{{{_fmt_cell(dt_mean, dt_std)}}} \\\\",
        "    \\midrule",
    ]

    baseline_order = ["Top OOB MCC", "Top OOB ACC", "Random"]

    for rep in baseline_order:
        cells = []
        for k in shared_values["subforest_sizes"]:
            row = _bl[(_bl["Representation"] == rep) & (_bl["Subforest Size"] == k)]
            if len(row) == 0:
                cells.append("--")
            else:
                _mcc = row["avg_mcc"].values[0]
                _std = row["avg_std"].values[0]
                _bold = _mcc == _best_overall_per_k.get(k)
                cells.append(_fmt_cell(_mcc, _std, bold=_bold))
        _latex_table1_lines.append(f"    {rep} & " + " & ".join(cells) + r" \\")
    _latex_table1_lines.append("    \\midrule")

    for _rep in REP_NAMES:
        _cells = []
        for k in shared_values["subforest_sizes"]:
            _row = _rep_k[
                (_rep_k["Representation"] == _rep) & (_rep_k["Subforest Size"] == k)
            ]
            if len(_row) == 0:
                _cells.append("--")
            else:
                _mcc = _row["avg_mcc"].values[0]
                _std = _row["avg_std"].values[0]
                _underline = _best_rep_per_k.get(k) == _rep
                _bold = _mcc == _best_overall_per_k.get(k)
                _cells.append(_fmt_cell(_mcc, _std, underline=_underline, bold=_bold))
        _display = _rep.replace("Topological Forest", "Topol. Forest")
        _latex_table1_lines.append(f"    {_display} & " + " & ".join(_cells) + " \\\\")
    _latex_table1_lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    _latex_table1 = "\n".join(_latex_table1_lines)
    suffix = "_recovery" if show_recovery else ""
    with open(
        f"{output_dir}/table_representation_vs_subforest_size{suffix}.txt", "w"
    ) as f:
        f.write(_latex_table1)
    print("Subforest Selection: Representation vs Subforest Size Table (LaTeX) - done.")
    print()


def print_config_vs_subforest_size(shared_values, output_dir, show_recovery):
    """Generate a LaTeX table summarizing the mean MCC, standard deviation, and significance percentage for each configuration and subforest size.

    Aggregation:
        MCC values are averaged across folds and datasets for each configuration and
        subforest size. Significance percentages are based on Wilcoxon tests of the
        fold-averaged dataset-level results.

    Args:
        shared_values (dict): Dictionary containing processed dataframes for different categories.
        output_dir (str): Directory where the LaTeX table will be saved.
    """
    shared_values["bl"] = _average_random_baseline(shared_values["bl"])
    if show_recovery:
        shared_values = _normalize_to_full_forest(shared_values, "MCC")

    _ff = (
        shared_values["ff"]
        .groupby(["Dataset"])["MCC"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    _ff_mean = _ff["mean"].mean()
    _ff_std = _ff["std"].mean()
    _dt = (
        shared_values["dt"]
        .groupby(["Dataset"])["MCC"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    _dt_mean = _dt["mean"].mean()
    _dt_std = _dt["std"].mean()
    _bl = (
        shared_values["bl"]
        .groupby(["Representation", "Dataset", "Subforest Size"])["MCC"]
        .agg(dataset_mcc="mean", dataset_std="std")
        .reset_index()
    )
    _bl = (
        _bl.groupby(["Representation", "Subforest Size"])
        .agg(avg_mcc=("dataset_mcc", "mean"), avg_std=("dataset_std", "mean"))
        .reset_index()
    )

    # For each k: pairwise Wilcoxon tests across all 40 configs to get sig%
    shared_values["rep"]["Config"] = (
        shared_values["rep"]["Representation"]
        + " + "
        + shared_values["rep"]["Selection Strategy"]
    )
    _configs = sorted(shared_values["rep"]["Config"].unique())

    # Pre-compute mean MCC per (Config, Dataset, k) for Wilcoxon testing
    _cfg_ds_k = (
        shared_values["rep"]
        .groupby(["Config", "Dataset", "Subforest Size"])["MCC"]
        .mean()
    )

    # Build sig matrix: sig_pct[cfg][k] = % of other configs this cfg beats
    _sig_pct = {cfg: {} for cfg in _configs}
    for k in shared_values["subforest_sizes"]:
        _vals_by_cfg = {}
        for cfg in _configs:
            try:
                _vals_by_cfg[cfg] = _cfg_ds_k.xs(
                    (cfg, k), level=("Config", "Subforest Size")
                ).values
            except KeyError:
                _vals_by_cfg[cfg] = None

        for cfg in _configs:
            if _vals_by_cfg[cfg] is None:
                _sig_pct[cfg][k] = np.nan
                continue
            _n_sig = 0
            _n_total = 0
            for other in _configs:
                if other == cfg or _vals_by_cfg[other] is None:
                    continue
                _diff = _vals_by_cfg[cfg] - _vals_by_cfg[other]
                if np.all(_diff == 0):
                    continue
                try:
                    _, _p = wilcoxon(_diff, alternative="greater")
                    _n_sig += int(_p < 0.05)
                except Exception:
                    pass
                _n_total += 1
            _sig_pct[cfg][k] = 100.0 * _n_sig / _n_total if _n_total > 0 else 0.0

    _cfg_ds_k_agg = (
        shared_values["rep"]
        .groupby(["Config", "Dataset", "Subforest Size"])["MCC"]
        .agg(dataset_mcc="mean", dataset_std="std")
        .reset_index()
    )
    _cfg_k_agg = (
        _cfg_ds_k_agg.groupby(["Config", "Subforest Size"])
        .agg(avg_mcc=("dataset_mcc", "mean"), avg_std=("dataset_std", "mean"))
        .reset_index()
    )

    # Sort configs by overall avg MCC descending for table order
    _cfg_order = (
        _cfg_k_agg.groupby("Config")["avg_mcc"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Best config per k (rep) by avg MCC
    _best_cfg_per_k_reps = (
        _cfg_k_agg.groupby("Subforest Size")
        .apply(lambda d: d.loc[d["avg_mcc"].idxmax(), "Config"])
        .to_dict()
    )

    # Best config per k (rep+bl) by avg MCC
    _all_k_agg = pd.concat(
        [
            _cfg_k_agg[["Subforest Size", "Config", "avg_mcc"]],
            _bl.rename(columns={"Representation": "Config"})[
                ["Subforest Size", "Config", "avg_mcc"]
            ],
        ],
        ignore_index=True,
    )
    _best_cfg_per_k = (
        _all_k_agg.groupby("Subforest Size")
        .apply(lambda d: d.loc[d["avg_mcc"].idxmax(), "Config"])
        .to_dict()
    )

    def _fmt_triple(mcc, std, sig=None, bold=False, underline=False):
        sig = None
        value = f"{mcc:.3f}"
        if underline:
            value = f"\\underline{{{value}}}"
        if bold:
            value = f"\\textbf{{{value}}}"
        if sig is None:
            return f"${value}\\pm{std:.3f}$"
        else:
            return f"${value}\\pm{std:.3f}$ / {sig:.0f}\\%"

    # Build a short config label: abbreviate representation
    def _abbrev_cfg(cfg):
        _rep_rename = {
            "Tree Descriptor": "TD",
            "Leaf Profile": "LP",
            "Feature Graph": "FG",
            "Topological Forest": "TF",
            "INDTree": "ID",
        }
        _strat_rename = {
            "agglomerative-performance": "agp",
            "agglomerative": "ag",
            "combination-genetic": "ge",
            "combination-greedy": "gr",
            "combination-simulated_annealing": "sa",
            "density": "de",
            "k-medoid-performance": "kmp",
            "k-medoid": "km",
        }
        rep, strat = cfg.split(" + ")
        rep = _rep_rename.get(rep, rep)
        strat = _strat_rename.get(strat, strat)
        return f"{rep}+{strat}"

    def _build_tabular(sf_sizes):
        _header = " & ".join([f"$k={k}$" for k in sf_sizes])

        lines = [
            f"  \\begin{{tabular}}{{l{'c' * len(sf_sizes)}}}",
            "    \\toprule",
            f"    Configuration & {_header} \\\\",
            "    \\midrule",
        ]

        lines += [
            f"    Full Forest & "
            f"\\multicolumn{{{len(sf_sizes)}}}{{c}}{{{_fmt_triple(_ff_mean, _ff_std)}}} \\\\",
            f"    Single DT & "
            f"\\multicolumn{{{len(sf_sizes)}}}{{c}}{{{_fmt_triple(_dt_mean, _dt_std)}}} \\\\",
            "    \\midrule",
        ]

        baseline_order = ["Top OOB MCC", "Top OOB ACC", "Random"]
        for baseline in baseline_order:
            cells = []
            for k in sf_sizes:
                row = _bl[
                    (_bl["Representation"] == baseline) & (_bl["Subforest Size"] == k)
                ]
                if len(row) == 0:
                    cells.append("--")
                else:
                    _bold = _best_cfg_per_k.get(k) == baseline
                    cells.append(
                        _fmt_triple(
                            row["avg_mcc"].values[0],
                            row["avg_std"].values[0],
                            bold=_bold,
                        )
                    )
            lines.append(f"    {baseline} & " + " & ".join(cells) + " \\\\")
        lines.append("    \\midrule")

        for cfg in _cfg_order:
            _cells = []
            for k in sf_sizes:
                _row = _cfg_k_agg[
                    (_cfg_k_agg["Config"] == cfg) & (_cfg_k_agg["Subforest Size"] == k)
                ]
                if len(_row) == 0:
                    _cells.append("--")
                else:
                    _bold = _best_cfg_per_k.get(k) == cfg
                    _underline = _best_cfg_per_k_reps.get(k) == cfg
                    _cells.append(
                        _fmt_triple(
                            _row["avg_mcc"].values[0],
                            _row["avg_std"].values[0],
                            _sig_pct[cfg][k],
                            _bold,
                            _underline,
                        )
                    )

            lines.append(f"    {_abbrev_cfg(cfg)} & " + " & ".join(_cells) + " \\\\")

        lines += [
            "    \\bottomrule",
            "  \\end{tabular}",
        ]

        return lines

    metric_name = "recovery relative to the full forest" if show_recovery else "MCC"
    _latex_table2_lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\caption{Per-configuration performance across subforest sizes. Each cell reports the mean "
        + metric_name
        + " values and their mean standard deviation. Bold: best overall value per $k$. Underline: best value among all configurations per $k$.}",
        "  \\label{tab:config_MCC}",
        "  \\resizebox{\\textwidth}{!}{%",
    ]

    _latex_table2_lines += _build_tabular(shared_values["subforest_sizes"])

    _latex_table2_lines += [
        "}",
        "\\end{table}",
    ]

    _latex_table2 = "\n".join(_latex_table2_lines)

    suffix = "_recovery" if show_recovery else ""
    with open(f"{output_dir}/table_config_vs_subforest_size{suffix}.txt", "w") as f:
        f.write(_latex_table2)
    print("Subforest Selection: Configuration vs Subforest Size Table (LaTeX) - done.")
    print()


def _average_random_baseline(df):
    df = df.copy()

    random = df[df["Representation"] == "Random"].copy()
    other = df[df["Representation"] != "Random"].copy()

    group_cols = [
        "Dataset",
        "Seed",
        "Fold",
        "Representation",
        "Selection Strategy",
        "Full Forest Size",
        "Subforest Size",
    ]

    def mean_array(series):
        arrays = np.stack(series.apply(ast.literal_eval).apply(np.asarray))
        return str(arrays.mean(axis=0).tolist())

    agg = {}

    for col in random.columns:
        if col in group_cols:
            continue
        elif col == "Feature Importances":
            agg[col] = mean_array
        elif pd.api.types.is_numeric_dtype(random[col]):
            agg[col] = "mean"
        else:
            agg[col] = "first"

    random = random.groupby(group_cols, dropna=False, as_index=False).agg(agg)

    return pd.concat([other, random], ignore_index=True)


def _normalize_to_full_forest(shared_values, metric="MCC"):
    """Return a copy of shared_values where the given metric is expressed
    relative to the corresponding Full Forest performance.

    Recovery = metric / Full Forest metric.

    Args:
        shared_values (dict): Dictionary containing processed dataframes for different categories.
        metric (str): The metric to normalize (default is "MCC").

    Returns:
        dict: A new dictionary with the same structure as shared_values, but with the specified metric normalized to the Full Forest performance.
    """
    normalized = deepcopy(shared_values)
    ff_lookup = (
        normalized["ff"].set_index(["Dataset", "Fold", "Seed"])[metric].to_dict()
    )
    for key in ["rep", "bl", "dt"]:
        df = normalized[key]
        full_metric = df.apply(
            lambda row: ff_lookup[(row["Dataset"], row["Fold"], row["Seed"])],
            axis=1,
        )
        valid = full_metric.notna() & np.isfinite(full_metric) & (full_metric != 0)
        df = df.loc[valid].copy()
        full_metric = full_metric.loc[valid]
        df[metric] = df[metric] / full_metric
        df = df[np.isfinite(df[metric])].copy()
        normalized[key] = df
    normalized["ff"][metric] = 1.0
    return normalized
