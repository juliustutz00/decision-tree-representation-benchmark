"""
Plotting and reporting functions for the perturbation benchmark.

This module provides functions to analyze the results of the perturbationbenchmark runs and generate plots and statistics.
It works with results files containing one or multiple datasets.
If multiple datasets are present, the results will be aggregated across datasets for the plots and statistics.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


def read_perturbation_benchmark_results(results_path, REP_NAMES, PERTURBATIONS):
    """Read the perturbation benchmark results from a CSV file and preprocess the data.

    Args:
        results_path (str): Path to the CSV file containing the perturbation benchmark results.
        REP_NAMES (list[str]): List of representation names to be used for similarity columns.

    Returns:
        pert_df (pd.DataFrame): Preprocessed DataFrame containing the perturbation benchmark results.
    """
    sim_cols = ["sim_" + r for r in REP_NAMES]

    pert_df = pd.read_csv(results_path, header=0)
    pert_df["dataset"] = pert_df["dataset"].ffill()

    pert_df = pert_df[pert_df["perturbation"].isin(PERTURBATIONS)].copy()

    num_cols = [
        "intensity",
        "performance_base",
        "performance_perturbed",
        "feature_importance_difference",
    ] + sim_cols
    for c in num_cols:
        pert_df[c] = pd.to_numeric(pert_df[c], errors="coerce")

    pert_df["mcc_diff"] = pert_df["performance_base"] - pert_df["performance_perturbed"]
    pert_df["mcc_diff_abs"] = pert_df["mcc_diff"].abs()

    return pert_df


def plot_rep_similarity_vs_performance_feature_importance(data, output_dir, REP_NAMES):
    """Plot the relationship between representation similarity and performance/feature importance difference for each representation.

    Args:
        data (pd.DataFrame): DataFrame containing the perturbation benchmark results.
        output_dir (str): Directory where the plots will be saved.
        REP_NAMES (list[str]): List of representation names to be considered.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    def _clean_xy(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    performance_data = []
    feature_importance_data = []

    for rep_name in REP_NAMES:
        fig_performance, ax_performance = plt.subplots(figsize=(6, 4))
        fig_feature_importance, ax_feature_importance = plt.subplots(figsize=(6, 4))

        x_raw = data[f"sim_{rep_name}"]
        y_raw_performance = np.asarray(
            data["performance_base"], dtype=float
        ) - np.asarray(data["performance_perturbed"], dtype=float)
        y_raw_feature_importance = np.asarray(
            data["feature_importance_difference"], dtype=float
        )

        x_all = 1.0 - np.asarray(x_raw, dtype=float)
        x_all_performance, y_all_performance = _clean_xy(x_all, y_raw_performance)
        x_all_feature_importance, y_all_feature_importance = _clean_xy(
            x_all, y_raw_feature_importance
        )

        mask_perf = np.isfinite(x_all) & np.isfinite(y_raw_performance)
        mask_feat = np.isfinite(x_all) & np.isfinite(y_raw_feature_importance)
        tmp_perf = pd.DataFrame(
            {
                "similarity": x_all_performance,
                "value": y_all_performance,
                "representation": rep_name,
                "type": "Performance",
                "perturbation": data.loc[mask_perf, "perturbation"].values,
                "intensity": data.loc[mask_perf, "intensity"].values,
            }
        )
        tmp_feat = pd.DataFrame(
            {
                "similarity": x_all_feature_importance,
                "value": y_all_feature_importance,
                "representation": rep_name,
                "type": "Feature Importance",
                "perturbation": data.loc[mask_feat, "perturbation"].values,
                "intensity": data.loc[mask_feat, "intensity"].values,
            }
        )
        performance_data.append(tmp_perf)
        feature_importance_data.append(tmp_feat)

        if "perturbation" in data.columns and "intensity" in data.columns:
            perturbations = sorted(
                data["perturbation"].dropna().unique().tolist(), key=lambda v: str(v)
            )
            intensities = data["intensity"].dropna().astype(float)
            min_i = float(intensities.min()) if len(intensities) else 0.0
            max_i = float(intensities.max()) if len(intensities) else 1.0

            cmap = plt.get_cmap("tab10")
            color_map = {p: cmap(i % 10) for i, p in enumerate(perturbations)}

            for p in perturbations:
                dpp = data[data["perturbation"] == p]
                uniq_ints = sorted(
                    [float(v) for v in dpp["intensity"].dropna().unique().tolist()]
                )
                labeled_perf = False
                labeled_feat = False
                for inten in uniq_ints:
                    dppi = dpp[dpp["intensity"].astype(float) == float(inten)]
                    xx_raw = dppi[f"sim_{rep_name}"]
                    yy_raw_performance = np.asarray(
                        dppi["performance_base"], dtype=float
                    ) - np.asarray(dppi["performance_perturbed"], dtype=float)
                    yy_raw_feature_importance = np.asarray(
                        dppi["feature_importance_difference"], dtype=float
                    )
                    xx = 1.0 - np.asarray(xx_raw, dtype=float)
                    xx_performance, yy_performance = _clean_xy(xx, yy_raw_performance)
                    xx_feature_importance, yy_feature_importance = _clean_xy(
                        xx, yy_raw_feature_importance
                    )
                    if not np.isfinite(float(inten)):
                        alpha = 0.15
                    elif max_i <= min_i:
                        alpha = 1.0
                    else:
                        t = (float(float(inten)) - float(min_i)) / (
                            float(max_i) - float(min_i)
                        )
                        t = float(np.clip(t, 0.0, 1.0))
                        alpha = 0.15 + t * (1.0 - 0.15)
                    if len(xx_performance) != 0:
                        ax_performance.scatter(
                            xx_performance,
                            yy_performance,
                            s=18,
                            color=color_map[p],
                            alpha=alpha,
                            label=str(p) if not labeled_perf else None,
                        )
                        labeled_perf = True
                    if len(xx_feature_importance) != 0:
                        ax_feature_importance.scatter(
                            xx_feature_importance,
                            yy_feature_importance,
                            s=18,
                            color=color_map[p],
                            alpha=alpha,
                            label=str(p) if not labeled_feat else None,
                        )
                        labeled_feat = True

        else:
            ax_performance.scatter(
                x_all_performance,
                y_all_performance,
                color="lightskyblue",
                alpha=0.7,
                s=18,
            )
            ax_feature_importance.scatter(
                x_all_feature_importance,
                y_all_feature_importance,
                color="mediumspringgreen",
                alpha=0.7,
                s=18,
            )

        if (
            len(x_all_performance) >= 2
            and np.std(x_all_performance) > 0
            and np.std(y_all_performance) > 0
        ):
            r_performance, pval_performance = pearsonr(
                x_all_performance, y_all_performance
            )
            m, b = np.polyfit(x_all_performance, y_all_performance, 1)
            xs = np.linspace(np.min(x_all_performance), np.max(x_all_performance), 200)
            ax_performance.plot(xs, m * xs + b, color="red", linewidth=2)
        else:
            r_performance, pval_performance = pearsonr(
                x_all_performance, y_all_performance
            )
        if (
            len(x_all_feature_importance) >= 2
            and np.std(x_all_feature_importance) > 0
            and np.std(y_all_feature_importance) > 0
        ):
            r_feature_importance, pval_feature_importance = pearsonr(
                x_all_feature_importance, y_all_feature_importance
            )
            m, b = np.polyfit(x_all_feature_importance, y_all_feature_importance, 1)
            xs = np.linspace(
                np.min(x_all_feature_importance), np.max(x_all_feature_importance), 200
            )
            ax_feature_importance.plot(xs, m * xs + b, color="red", linewidth=2)
        else:
            r_feature_importance, pval_feature_importance = pearsonr(
                x_all_feature_importance, y_all_feature_importance
            )

        ax_performance.set_xlabel("Δ Representation Similarity")
        ax_performance.set_ylabel("Δ MCC")
        ax_performance.set_title(
            f"Similarity vs Performance — {rep_name}\n"
            f"n={len(x_all)}, r={r_performance:.2f}, p={pval_performance:.3f}"
        )
        fig_performance.tight_layout()
        if "perturbation" in data.columns:
            ax_performance.text(
                0.98,
                0.02,
                "lighter = lower intensity\ndarker = higher intensity",
                transform=ax_performance.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )

            leg = ax_performance.legend(fontsize=8, frameon=False)
            for h in leg.legend_handles:
                h.set_alpha(1.0)
        fig_performance.savefig(
            f"{output_dir}/fig_similarity_vs_performance_{rep_name}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig_performance)

        ax_feature_importance.set_xlabel("Δ Representation Similarity")
        ax_feature_importance.set_ylabel("Δ Feature Importance")
        ax_feature_importance.set_title(
            f"Similarity vs Feature Importance Difference — {rep_name}\n"
            f"n={len(x_all_feature_importance)}, r={r_feature_importance:.2f}, p={pval_feature_importance:.3f}"
        )
        fig_feature_importance.tight_layout()
        if "perturbation" in data.columns:
            ax_feature_importance.text(
                0.98,
                0.02,
                "lighter = lower intensity\ndarker = higher intensity",
                transform=ax_feature_importance.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )

            leg = ax_feature_importance.legend(fontsize=8, frameon=False)
            for h in leg.legend_handles:
                h.set_alpha(1.0)

        fig_feature_importance.savefig(
            f"{output_dir}/fig_similarity_vs_feature_importance_{rep_name}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig_feature_importance)

    df_plot = pd.concat(performance_data + feature_importance_data, ignore_index=True)
    palette = dict(
        zip(
            sorted(df_plot["perturbation"].unique()),
            sns.color_palette("tab10", df_plot["perturbation"].nunique()),
        )
    )
    g = sns.relplot(
        data=df_plot,
        x="similarity",
        y="value",
        col="representation",
        row="type",
        hue="perturbation",
        palette=palette,
        kind="scatter",
        height=3.5,
        aspect=1,
        facet_kws={"sharex": True, "sharey": False},
    )
    for (plot_type, rep_name), ax in g.axes_dict.items():
        subset = df_plot[
            (df_plot["type"] == plot_type) & (df_plot["representation"] == rep_name)
        ]
        sns.regplot(
            data=subset,
            x="similarity",
            y="value",
            scatter=False,
            ci=None,
            color="red",
            line_kws={"lw": 2},
            ax=ax,
        )
    for (plot_type, rep_name), ax in g.axes_dict.items():
        subset = df_plot[
            (df_plot["type"] == plot_type) & (df_plot["representation"] == rep_name)
        ]

        x = subset["similarity"].to_numpy()
        y = subset["value"].to_numpy()

        if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
            r, p = pearsonr(x, y)
            stats = f"r = {r:.2f}, p = {p:.3f}"
        else:
            stats = "r = –, p = –"
        if plot_type == "Performance":
            ax.set_title(f"{rep_name}\n{stats}")
        else:
            ax.set_title(stats)
    for ax in g.axes.flat:
        ax.set_ylabel("")
    g.axes[0, 0].set_ylabel("Δ MCC")
    g.axes[1, 0].set_ylabel("Δ Feature Importance")
    for ax in g.axes[0]:
        ax.set_xlabel("")
    for ax in g.axes[1]:
        ax.set_xlabel("Δ Representation Similarity")
    g.figure.savefig(
        f"{output_dir}/similarity_combined.png", dpi=600, bbox_inches="tight"
    )

    print("Perturbation Benchmark: Correlation scatter plots - done.")
    print()


def plot_similarity_vs_intensity_per_perturbation(
    data, output_dir, REP_NAMES, PERTURBATIONS
):
    """Plot the relationship between representation similarity and perturbation intensity for each perturbation type.

    Args:
        data (pd.DataFrame): DataFrame containing the perturbation benchmark results.
        output_dir (str): Directory where the plots will be saved.
        REP_NAMES (list[str]): List of representation names to be considered.
        PERTURBATIONS (list[str]): List of perturbation types to be considered.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("colorblind")

    sim_cols = ["sim_" + r for r in REP_NAMES]

    perturbations_present = [
        p for p in PERTURBATIONS if p in data["perturbation"].dropna().unique()
    ]
    perturbation_labels = {
        p: p.replace("_", " ").title() for p in perturbations_present
    }

    _agg = (
        data.groupby(["perturbation", "intensity"])[
            sim_cols + ["mcc_diff_abs", "feature_importance_difference"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    _agg.columns = ["perturbation", "intensity"] + [
        f"{c}_{s}" for c, s in _agg.columns[2:]
    ]

    fig_sim_intensity, axes = plt.subplots(
        1, len(perturbations_present), figsize=(15, 3.8), sharey=True
    )
    for ax, pert in zip(axes, perturbations_present):
        sub = _agg[_agg["perturbation"] == pert].sort_values("intensity")
        for i, (col, name) in enumerate(zip(sim_cols, REP_NAMES)):
            y_col = f"{col}_mean"
            ax.plot(
                sub["intensity"],
                sub[y_col],
                marker="o",
                markersize=4,
                color=palette[i],
                label=name,
                linewidth=1.6,
            )
        ax.set_title(perturbation_labels[pert], fontsize=11)
        ax.set_xlabel("Intensity", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Mean Similarity", fontsize=10)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig_sim_intensity.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.18),
        fontsize=10,
        frameon=True,
    )
    fig_sim_intensity.suptitle(
        "Mean Representation Similarity vs. Perturbation Intensity", fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/fig_similarity_vs_intensity_per_perturbations.png",
        dpi=600,
        bbox_inches="tight",
    )
    print("Perturbation Benchmark: Similarity vs Intensity per Perturbation – done.")
    print()
