import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)
from scipy.stats import pearsonr, spearmanr


"""
Utility functions for random-forest training, evaluation, similarity conversion,
and subforest selection.
"""

def train_own_random_forest(X_train, y_train, n_trees, print_progress, seed):
    """Train bootstrap-sampled decision trees and return OOB bookkeeping."""
    random_forest_trees = []
    bootstrap_indices_list = []
    oob_indices_list = []
    for tree in range(n_trees):
        bootstrap_indices = generate_sample_indices(
            seed + tree, X_train.shape[0], X_train.shape[0]
        )
        X_boot, y_boot = X_train[bootstrap_indices], y_train[bootstrap_indices]
        oob_indices = generate_unsampled_indices(X_train.shape[0], bootstrap_indices)
        template_tree = DecisionTreeClassifier(max_depth=10, max_features="sqrt", random_state=seed)
        template_tree.fit(X_boot, y_boot)
        random_forest_trees.append(template_tree)
        bootstrap_indices_list.append(bootstrap_indices)
        oob_indices_list.append(oob_indices)
    return random_forest_trees, bootstrap_indices_list, oob_indices_list

def get_combined_correlation(df, pearson_weight=0.5, spearman_weight=0.5):
    """Compute pairwise representation correlation table across all `sim_` columns."""
    if pearson_weight + spearman_weight != 1:
        raise ValueError("(pearson_weight + spearman_weight) has to be equal to 1.")
    
    sim_columns = [col for col in df.columns if col.startswith('sim_')]
    results = []
    for i, col1 in enumerate(sim_columns):
        for col2 in sim_columns[i+1:]:
            pearson_corr, spearman_corr, combined_corr = __combined_pearson_spearman_corr(df[col1], df[col2], pearson_weight, spearman_weight)
            results.append({
                'Representation 1': col1,
                'Representation 2': col2,
                'Pearson Correlation': pearson_corr,
                'Spearman Correlation': spearman_corr,
                'Combined Correlation': combined_corr
            })
    df_correlation = pd.DataFrame(results)
    return df_correlation

def generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    random_instance = np.random.RandomState(random_state)
    sample_indices = random_instance.randint(low=0, high=n_samples, size=n_samples_bootstrap, dtype=np.int32)
    return sample_indices

def generate_unsampled_indices(n_samples, sample_indices):
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]
    return unsampled_indices

def evaluate_forest(X_test, y_test, trees, n_instances_test, n_classes): 
    """
    Evaluate an ensemble by majority vote + averaged probabilities.

    Returns hard predictions, probabilities, and standard classification metrics.
    """
    all_preds = np.vstack([t.predict(X_test) for t in trees])
    hard_preds = np.zeros(n_instances_test, dtype=int)

    for i in range(n_instances_test):
        counts = np.bincount(all_preds[:, i].astype(int), minlength=n_classes)
        hard_preds[i] = counts.argmax()

    probas = np.mean(
        [t.predict_proba(X_test) for t in trees],
        axis=0
    )

    metrics = {
        "accuracy": accuracy_score(y_test, hard_preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, hard_preds),
        "macro_f1": f1_score(y_test, hard_preds, average="macro"),
    }

    try:
        if n_classes == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, probas[:, 1])
        else:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_test, probas, multi_class="ovr"
            )
    except ValueError:
        metrics["roc_auc"] = np.nan

    return {
        "hard_predictions": hard_preds,
        "probabilities": probas,
        "metrics": metrics,
    }

def prediction_agreement(preds_a, preds_b):
    return np.mean(preds_a == preds_b)

def similarity_to_distance_matrix(similarity_fn, n):
    """Build symmetric distance matrix by min-max normalizing pairwise similarity."""
    similarity_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            sim = similarity_fn(i, j)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    sim_min, sim_max = similarity_matrix.min(), similarity_matrix.max()
    similarity_matrix_normalized = (similarity_matrix - sim_min) / (sim_max - sim_min + 1e-10)
    distance_matrix = 1.0 - similarity_matrix_normalized
    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix

def select_subforest_via_clustering(distance_matrix, subforest_size, clustering_method, seed):
    """Select representative tree indices via k-medoids, agglomerative, or density."""
    if clustering_method == "k-medoid":
        kmed_clustering = KMedoids(
            n_clusters=subforest_size, metric="precomputed", random_state=seed
        )
        kmed_clustering.fit(distance_matrix)
        subforest_indices = [int(x) for x in kmed_clustering.medoid_indices_]
    elif clustering_method == "agglomerative":
        agglomerative_clustering = AgglomerativeClustering(
            n_clusters=subforest_size, metric="precomputed", linkage="average"
        )
        labels = agglomerative_clustering.fit_predict(distance_matrix)
        subforest_indices = []
        for cluster_id in range(subforest_size):
            cluster_indices = np.where(labels == cluster_id)[0]
            if cluster_indices.size == 0:
                continue
            cluster_distances = distance_matrix[
                np.ix_(cluster_indices, cluster_indices)
            ]
            medoid_index_within_cluster = cluster_indices[
                int(np.argmin(cluster_distances.sum(axis=1)))
            ]
            subforest_indices.append(medoid_index_within_cluster)
    elif clustering_method == "density":
        subforest_indices = select_subforest_via_density(distance_matrix, subforest_size, seed, 1.0, 1.5)
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    return subforest_indices

def select_subforest_via_density(distance_matrix, subforest_size, seed, sigma=1.0, alpha=1.5):
    # future work: try to compute density using parzen windows
    np.random.seed(seed)
    n_trees = distance_matrix.shape[0]
    densities = np.exp(-distance_matrix**2 / (2 * sigma**2)).sum(axis=1)
    probabilities = densities / densities.sum()
    remaining = set(range(n_trees))
    
    selected = []
    for _ in range(subforest_size):
        probs = np.array([probabilities[i] for i in remaining])
        probs = probs / probs.sum()
        
        chosen = np.random.choice(list(remaining), p=probs)
        selected.append(chosen)
        remaining.remove(chosen)
        
        for i in remaining:
            probabilities[i] *= distance_matrix[i, chosen]**alpha
    
    return selected

def compute_similarity_to_base_tree(similarity_fn, len_tree_representations, len_perturbations, len_intensities, len_strengths, len_perturbation_runs):
    """Map perturbed-tree indices to similarity against their corresponding base tree."""
    similarity_values = []
    perturbation_offset = len_perturbations * len_intensities * len_strengths * len_perturbation_runs + 1
    for idx in range(len_tree_representations):
        if (idx % perturbation_offset) != 0:
            base_tree_idx = (idx // perturbation_offset) * perturbation_offset
            similarity = similarity_fn(idx, base_tree_idx)
            similarity_values.append(
                {"tree_idx": idx - 1, "similarity_to_base": similarity}
            )
    return similarity_values


def __combined_pearson_spearman_corr(col1, col2, pearson_weight, spearman_weight):
    pearson_corr, _ = pearsonr(col1, col2)
    spearman_corr, _ = spearmanr(col1, col2)
    combined_corr = (pearson_corr * pearson_weight) + (spearman_corr * spearman_weight)
    return pearson_corr, spearman_corr, combined_corr
