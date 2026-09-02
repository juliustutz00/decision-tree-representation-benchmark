"""
Plotting and reporting functions for the DTRBench analysis.

This module orchestrates functions to analyze the results of the benchmark runs and generate plots and statistics.
It works with results files containing one or multiple datasets.
If multiple datasets are present, the results will be aggregated across datasets for the plots and statistics.
"""

import warnings

from dtrbench.analysis.perturbation_results import (
    plot_rep_similarity_vs_performance_feature_importance,
    plot_similarity_vs_intensity_per_perturbation,
    read_perturbation_benchmark_results,
)
from dtrbench.analysis.resource_results import (
    plot_memory_analysis,
    plot_runtime_analysis,
    read_resource_benchmark_results,
)
from dtrbench.analysis.subforest_results import (
    plot_kendalls_w_vs_config,
    plot_mcc_boxplots,
    plot_mcc_representation_selection_strategy,
    plot_rf_compression,
    plot_spearman_vs_subforest_size,
    plot_std_representation_selection_strategy,
    print_config_vs_subforest_size,
    print_representation_vs_subforest_size,
    read_subforest_benchmark_result,
)
from dtrbench.config.loader import load_analysis_config

warnings.filterwarnings("ignore")


def report(config_path):
    """
    Analyze the results of the benchmark runs and generate plots and statistics.

    Works with results files containing one or multiple datasets. If multiple datasets are present, the results will be aggregated across datasets for the plots and statistics.

    Args:
        config_path (str): Path to the report configuration file.
    """

    config = load_analysis_config(config_path)

    output_dir = config["output_dir"]
    perturbation_benchmark_results_path = config.get(
        "perturbation_benchmark_results_path"
    )
    subforest_benchmark_results_path = config.get("subforest_benchmark_results_path")
    resource_benchmark_represent_results_path = config.get(
        "resource_benchmark_represent_results_path"
    )
    resource_benchmark_similarity_results_path = config.get(
        "resource_benchmark_similarity_results_path"
    )
    rep_similarity_vs_performance_feature_importance = config.get(
        "rep_similarity_vs_performance_feature_importance", True
    )
    similarity_vs_intensity_per_perturbation = config.get(
        "similarity_vs_intensity_per_perturbation", True
    )
    rf_compression = config.get("rf_compression", True)
    mcc_boxplots = config.get("mcc_boxplots", True)
    mcc_representation_selection_strategy = config.get(
        "mcc_representation_selection_strategy", True
    )
    std_representation_selection_strategy = config.get(
        "std_representation_selection_strategy", True
    )
    kendalls_w_vs_config = config.get("kendalls_w_vs_config", True)
    spearman_vs_subforest_size = config.get("spearman_vs_subforest_size", True)
    representation_vs_subforest_size = config.get(
        "representation_vs_subforest_size", True
    )
    config_vs_subforest_size = config.get("config_vs_subforest_size", True)
    resource_benchmark_represent = config.get("resource_benchmark_represent", True)
    resource_benchmark_similarity = config.get("resource_benchmark_similarity", True)

    rep_names = config.get(
        "representations",
        [
            "Tree Descriptor",
            "Leaf Profile",
            "Feature Graph",
            "Topological Forest",
            "INDTree",
        ],
    )
    perturbations = config.get(
        "perturbations",
        [
            "change_threshold",
            "change_feature",
            "swap_nodes",
            "remove_nodes",
            "add_nodes",
        ],
    )
    subforest_sizes = config.get("subforest_sizes", [2, 5, 10, 15, 20, 25, 30])
    selection_strategies = config.get(
        "selection_strategies",
        [
            "k-medoid",
            "k-medoid-performance",
            "agglomerative",
            "agglomerative-performance",
            "density",
            "combination-greedy",
            "combination-genetic",
            "combination-simulated_annealing",
        ],
    )
    show_recovery = config.get("show_recovery", False)

    if perturbation_benchmark_results_path is not None:
        perturbation_data = read_perturbation_benchmark_results(
            perturbation_benchmark_results_path, rep_names, perturbations
        )
        if rep_similarity_vs_performance_feature_importance:
            plot_rep_similarity_vs_performance_feature_importance(
                perturbation_data, output_dir, rep_names
            )
        if similarity_vs_intensity_per_perturbation:
            plot_similarity_vs_intensity_per_perturbation(
                perturbation_data, output_dir, rep_names, perturbations
            )

    if subforest_benchmark_results_path is not None:
        subforest_data = read_subforest_benchmark_result(
            subforest_benchmark_results_path,
            rep_names,
            subforest_sizes,
            selection_strategies,
        )
        if rf_compression:
            plot_rf_compression(
                subforest_data.copy(),
                output_dir,
                rep_names,
                selection_strategies,
                show_recovery,
            )
        if mcc_boxplots:
            plot_mcc_boxplots(
                subforest_data.copy(),
                output_dir,
                rep_names,
                selection_strategies,
                show_recovery,
            )
        if mcc_representation_selection_strategy:
            plot_mcc_representation_selection_strategy(
                subforest_data.copy(),
                output_dir,
                rep_names,
                selection_strategies,
                show_recovery,
            )
        if std_representation_selection_strategy:
            plot_std_representation_selection_strategy(
                subforest_data.copy(),
                output_dir,
                rep_names,
                selection_strategies,
                show_recovery,
            )
        if kendalls_w_vs_config:
            plot_kendalls_w_vs_config(
                subforest_data.copy(),
                output_dir,
                rep_names,
                selection_strategies,
                show_recovery,
            )
        if spearman_vs_subforest_size:
            plot_spearman_vs_subforest_size(
                subforest_data.copy(),
                output_dir,
                rep_names,
                selection_strategies,
                show_recovery,
            )
        if representation_vs_subforest_size:
            print_representation_vs_subforest_size(
                subforest_data.copy(), output_dir, rep_names, show_recovery
            )
        if config_vs_subforest_size:
            print_config_vs_subforest_size(
                subforest_data.copy(), output_dir, show_recovery
            )

    if (
        resource_benchmark_represent_results_path is not None
        or resource_benchmark_similarity_results_path is not None
    ):
        resource_benchmark_represent_data = read_resource_benchmark_results(
            resource_benchmark_represent_results_path
        )
        resource_benchmark_similarity_data = read_resource_benchmark_results(
            resource_benchmark_similarity_results_path
        )

        if (
            resource_benchmark_represent_results_path is not None
            and resource_benchmark_represent
        ):
            (
                plot_runtime_analysis(
                    resource_benchmark_represent_data, output_dir, rep_names
                )
                if resource_benchmark_represent_data is not None
                else None
            )
            (
                plot_runtime_analysis(
                    resource_benchmark_similarity_data, output_dir, rep_names
                )
                if resource_benchmark_similarity_data is not None
                else None
            )
        if (
            resource_benchmark_similarity_results_path is not None
            and resource_benchmark_similarity
        ):
            (
                plot_memory_analysis(
                    resource_benchmark_represent_data, output_dir, rep_names
                )
                if resource_benchmark_represent_data is not None
                else None
            )
            (
                plot_memory_analysis(
                    resource_benchmark_similarity_data, output_dir, rep_names
                )
                if resource_benchmark_similarity_data is not None
                else None
            )
