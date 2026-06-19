import numpy as np
import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score, 
    matthews_corrcoef,
    average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize
from scipy.stats import pearsonr, spearmanr


def train_own_random_forest(X_train, y_train, n_trees, print_progress, seed):
    random_forest_trees = []
    bootstrap_indices_list = []
    oob_indices_list = []
    for tree in range(n_trees):
        bootstrap_indices = generate_sample_indices(
            seed + tree, X_train.shape[0], X_train.shape[0]
        )
        X_boot, y_boot = X_train[bootstrap_indices], y_train[bootstrap_indices]
        oob_indices = generate_unsampled_indices(X_train.shape[0], bootstrap_indices)
        X_oob, y_oob = X_train[oob_indices], y_train[oob_indices]
        template_tree = DecisionTreeClassifier(max_depth=6, max_features="sqrt", random_state=seed)
        template_tree.fit(X_boot, y_boot)
        if print_progress and False:
            print(
                "Performance of unperturbed tree on OOB data:",
                template_tree.score(X_oob, y_oob),
            )
        random_forest_trees.append(template_tree)
        bootstrap_indices_list.append(bootstrap_indices)
        oob_indices_list.append(oob_indices)
    return random_forest_trees, bootstrap_indices_list, oob_indices_list

def get_combined_correlation(df, pearson_weight=0.5, spearman_weight=0.5):
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

def _safe_predict_proba(tree, X, n_classes):
    probas = tree.predict_proba(X)
    
    if probas.shape[1] == n_classes:
        return probas
    
    full_probas = np.zeros((X.shape[0], n_classes), dtype=probas.dtype)
    
    for k, class_label in enumerate(tree.classes_):
        if class_label < n_classes:
            full_probas[:, int(class_label)] = probas[:, k]
            
    return full_probas

def evaluate_forest(X_test, y_test, trees, n_instances_test, n_classes): 
    all_preds = np.vstack([t.predict(X_test) for t in trees])
    hard_preds = np.zeros(n_instances_test, dtype=int)

    for i in range(n_instances_test):
        counts = np.bincount(all_preds[:, i].astype(int), minlength=n_classes)
        hard_preds[i] = counts.argmax()

    probas_list = [_safe_predict_proba(t, X_test, n_classes) for t in trees]
    probas = np.mean(probas_list, axis=0)

    metrics = {
        "accuracy": accuracy_score(y_test, hard_preds),
        "macro_f1": f1_score(y_test, hard_preds, average="macro"),
        "mcc": matthews_corrcoef(y_test, hard_preds),
    }

    # ROC AUC
    try:
        if n_classes == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, probas[:, 1])
        else:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_test, probas, multi_class="ovr"
            )
    except ValueError:
        if n_classes == 2:
            metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc_ovr"] = np.nan

    # PR AUC
    try:
        if n_classes == 2:
            metrics["pr_auc"] = average_precision_score(y_test, probas[:, 1])
        else:
            y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
            metrics["pr_auc_ovr"] = average_precision_score(
                y_test_bin, probas, average="macro"
            )
    except ValueError:
        if n_classes == 2:
            metrics["pr_auc"] = np.nan
        else:
            metrics["pr_auc_ovr"] = np.nan

    # minority class metrics (one-vs-rest)
    y_test_int = np.asarray(y_test).astype(int)
    class_counts = np.bincount(y_test_int, minlength=n_classes)
    present_classes = np.where(class_counts > 0)[0]

    if present_classes.size > 0:
        minority_class = int(present_classes[np.argmin(class_counts[present_classes])])
        metrics["minority_class"] = minority_class
        metrics["minority_support"] = int(class_counts[minority_class])

        metrics["minority_precision"] = float(
            precision_score(
                y_test_int,
                hard_preds,
                labels=[minority_class],
                average=None,
                zero_division=0
            )[0]
        )
        metrics["minority_recall"] = float(
            recall_score(
                y_test_int,
                hard_preds,
                labels=[minority_class],
                average=None,
                zero_division=0
            )[0]
        )
        metrics["minority_f1"] = float(
            f1_score(
                y_test_int,
                hard_preds,
                labels=[minority_class],
                average=None,
                zero_division=0
            )[0]
        )
    else:
        metrics["minority_class"] = np.nan
        metrics["minority_support"] = np.nan
        metrics["minority_precision"] = np.nan
        metrics["minority_recall"] = np.nan
        metrics["minority_f1"] = np.nan

    # feature importances
    n_features = X_test.shape[1]
    importances = []
    for t in trees:
        if hasattr(t, "feature_importances_"):
            fi = np.asarray(t.feature_importances_, dtype=float)
            if fi.shape[0] != n_features:
                fi = np.pad(fi, (0, max(0, n_features - fi.shape[0])), mode="constant")
            importances.append(fi[:n_features])
        else:
            importances.append(np.zeros(n_features, dtype=float))

    if len(importances) > 0:
        feature_importances = np.mean(importances, axis=0)
    else:
        feature_importances = np.zeros(n_features, dtype=float)

    return {
        "hard_predictions": hard_preds,
        "probabilities": probas,
        "metrics": metrics,
        "feature_importances": feature_importances
    }

def prediction_agreement(preds_a, preds_b):
    return np.mean(preds_a == preds_b)

def similarity_to_distance_matrix(similarity_fn, n):
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

def compute_similarity_to_base_tree(similarity_fn, len_tree_representations, len_perturbations, len_intensities, len_strengths, len_perturbation_runs):
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

def tree_metric_score(tree, X, y, metric="accuracy", average="macro"):
    if X is None or y is None or len(y) == 0:
        return np.nan

    y_pred = tree.predict(X)

    if metric == "accuracy":
        return float(np.mean(y_pred == y))
    if metric == "f1":
        return float(f1_score(y, y_pred, average=average, zero_division=0))
    if metric == "mcc":
        return float(matthews_corrcoef(y, y_pred))

    raise ValueError(f"Unsupported tree metric: {metric}")


def shared_metric_cols(eval_result):
        m = eval_result["metrics"]
        return {
            "acc": m.get("accuracy", np.nan),
            "macro_f1": m.get("macro_f1", np.nan),
            "mcc": m.get("mcc", np.nan),
            "roc_auc": m.get("roc_auc", m.get("roc_auc_ovr", np.nan)),
            "pr_auc": m.get("pr_auc", m.get("pr_auc_ovr", np.nan)),
            "minority_class": m.get("minority_class", np.nan),
            "minority_support": m.get("minority_support", np.nan),
            "minority_precision": m.get("minority_precision", np.nan),
            "minority_recall": m.get("minority_recall", np.nan),
            "minority_f1": m.get("minority_f1", np.nan),
            "feature_importances": json.dumps([float(f"{x:.10f}") for x in np.asarray(eval_result.get("feature_importances", np.array([])))])
        }
