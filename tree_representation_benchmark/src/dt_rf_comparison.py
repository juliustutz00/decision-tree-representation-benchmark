import io
import sys
import re
import os
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from ucimlrepo import fetch_ucirepo, list_available_datasets


def _export_class_distributions_to_excel(dataset_ids,out_xlsx_path,*,drop_missing_targets=True):
    rows = []

    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(out_xlsx_path) or ".", exist_ok=True)

    for dataset_id in dataset_ids:
        try:
            ds = fetch_ucirepo(id=int(dataset_id))
            dataset_name = ds.metadata.get("name", f"id_{dataset_id}")

            y = ds.data.targets

            # Skip datasets without targets
            if y is None:
                print(f"Skipping dataset {dataset_name} (ID {dataset_id}): no target found.")
                continue

            # Targets can be a Series or DataFrame
            if isinstance(y, pd.DataFrame):
                if y.shape[1] == 0:
                    print(f"Skipping dataset {dataset_name} (ID {dataset_id}): empty target DataFrame.")
                    continue
                y_series = y.iloc[:, 0]
            else:
                y_series = pd.Series(y)

            # Remove missing targets if requested
            if drop_missing_targets:
                y_series = y_series.dropna()

            # Skip empty targets
            if len(y_series) == 0:
                print(f"Skipping dataset {dataset_name} (ID {dataset_id}): target empty after dropna().")
                continue

            # Compute normalized class distribution
            vc = y_series.value_counts(normalize=True, dropna=False)

            class_distribution = [
                round(float(p), 4) for p in vc.tolist()
            ]

            rows.append({
                "Dataset ID": int(dataset_id),
                "Dataset Name": dataset_name,
                "Class Distribution": str(class_distribution)
            })

        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {e}")

    # Create final DataFrame
    df = pd.DataFrame(rows).sort_values(
        ["Dataset Name", "Dataset ID"],
        ascending=[True, True]
    )

    # Export to Excel
    df.to_excel(out_xlsx_path, index=False)

    print(f"Done. Exported to: {out_xlsx_path}")

def _analyze_results(result_path):
    files = {
        "f1": "results_f1.xlsx",
        "mcc": "results_mcc.xlsx",
        "roc": "results_roc_auc.xlsx",
    }

    dfs = {}
    for key, filename in files.items():
        path = os.path.join(result_path, filename)
        if not os.path.exists(path):
            print(f"Datei nicht gefunden: {path}")
            return

        df = pd.read_excel(path)
        df = df.copy()
        df[f"rank_{key}"] = range(1, len(df) + 1)
        dfs[key] = df

    merged = dfs["f1"].copy()

    # add ranks from the other metrics
    merged = merged.merge(
        dfs["mcc"][["dataset_name", "rank_mcc"]],
        on="dataset_name",
        how="left"
    ).merge(
        dfs["roc"][["dataset_name", "rank_roc"]],
        on="dataset_name",
        how="left"
    )

    merged["rank_sum"] = merged["rank_f1"] + merged["rank_mcc"] + merged["rank_roc"]
    merged = merged.sort_values(["rank_sum", "dataset_name"], ascending=[True, True]).reset_index(drop=True)
    info_cols_preferred = ["dataset_name", "n_instances", "n_features", "n_classes", "has_missing_values"]
    rank_cols = ["rank_f1", "rank_mcc", "rank_roc", "rank_sum"]

    info_cols = [c for c in info_cols_preferred if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in info_cols + rank_cols]
    final_cols = info_cols + rank_cols + other_cols
    merged = merged[final_cols]

    merged.to_excel(os.path.join(result_path, "results_merged.xlsx"), index=False)

def _create_results(results, savepath):
    df = pd.DataFrame(results)
    os.makedirs(savepath, exist_ok=True)
    df["f1_diff"] = df["f1_rf"] - df["f1_dt"]
    df["mcc_diff"] = df["mcc_rf"] - df["mcc_dt"]
    df["roc_diff"] = df["roc_rf"] - df["roc_dt"]
    df_f1 = df.sort_values("f1_diff", ascending=False)
    df_mcc = df.sort_values("mcc_diff", ascending=False)
    df_roc = df.sort_values("roc_diff", ascending=False)
    df_f1.to_csv(savepath + "results_f1.csv", index=False)
    df_mcc.to_csv(savepath + "results_mcc.csv", index=False)
    df_roc.to_csv(savepath + "results_roc_auc.csv", index=False)

def _create_dt_rf_comparison(classification_ids):
    results = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for dataset_id in classification_ids:
        try:
            ds = fetch_ucirepo(id=dataset_id)
            X = ds.data.features
            y = ds.data.targets
            dataset_name = ds.metadata.get("name", f"id_{dataset_id}")

            # only numerical features
            X = X.select_dtypes(include=[np.number])

            # if no features left, skip
            if X.shape[1] == 0:
                continue

            # missing values info
            has_missing = ds.metadata.get("has_missing_values", "no")

            # drop rows with missing target values
            data = pd.concat([X, y], axis=1).dropna()
            X = data[X.columns]
            y = data[y.columns[0]]

            # number of classes
            n_classes = y.nunique()

            # ROC AUC: binary vs. multi-class
            is_binary = n_classes == 2

            min_class_count = y.value_counts().min()
            if min_class_count < 10:
                print(f"Dataset {dataset_name} skipped: smallest class has only {min_class_count} sample(s).")
                continue

            # not enough samples for CV
            if len(X) < 20:
                continue

            # CV-Scores 
            dt_scores = defaultdict(list)
            rf_scores = defaultdict(list)

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # models
                dt = DecisionTreeClassifier(max_depth=10, random_state=42)
                rf = RandomForestClassifier(max_depth=10, n_estimators=250, random_state=42)

                dt.fit(X_train, y_train)
                rf.fit(X_train, y_train)

                # predictions
                y_pred_dt = dt.predict(X_test)
                y_pred_rf = rf.predict(X_test)

                # probabilities for ROC AUC
                try:
                    y_proba_dt = dt.predict_proba(X_test)
                    y_proba_rf = rf.predict_proba(X_test)
                except:
                    continue

                # F1
                f1_dt = f1_score(y_test, y_pred_dt, average="weighted")
                f1_rf = f1_score(y_test, y_pred_rf, average="weighted")

                # MCC
                mcc_dt = matthews_corrcoef(y_test, y_pred_dt)
                mcc_rf = matthews_corrcoef(y_test, y_pred_rf)

                # ROC AUC
                try:
                    if is_binary:
                        roc_dt = roc_auc_score(y_test, y_proba_dt[:, 1])
                        roc_rf = roc_auc_score(y_test, y_proba_rf[:, 1])
                    else:
                        roc_dt = roc_auc_score(y_test, y_proba_dt, multi_class="ovr")
                        roc_rf = roc_auc_score(y_test, y_proba_rf, multi_class="ovr")
                except:
                    continue

                dt_scores["f1"].append(f1_dt)
                dt_scores["mcc"].append(mcc_dt)
                dt_scores["roc"].append(roc_dt)

                rf_scores["f1"].append(f1_rf)
                rf_scores["mcc"].append(mcc_rf)
                rf_scores["roc"].append(roc_rf)

            # if no valid scores, skip
            if len(dt_scores["f1"]) == 0:
                continue

            # average scores
            result = {
                "dataset_name": dataset_name,
                "n_features": X.shape[1],
                "n_instances": len(X),
                "n_classes": n_classes,
                "has_missing_values": has_missing,

                "f1_dt": np.mean(dt_scores["f1"]),
                "mcc_dt": np.mean(dt_scores["mcc"]),
                "roc_dt": np.mean(dt_scores["roc"]),

                "f1_rf": np.mean(rf_scores["f1"]),
                "mcc_rf": np.mean(rf_scores["mcc"]),
                "roc_rf": np.mean(rf_scores["roc"]),
            }

            results.append(result)

        except Exception as e:
            print(f"Error with Dataset {dataset_id}: {e}")

    return results

# filter all available dataset IDs and query only those that are classification datasets
def _filter_classification_ids(dataset_id):
    buffer = io.StringIO()
    sys.stdout = buffer
    list_available_datasets(filter="python")
    sys.stdout = sys.__stdout__
    output = buffer.getvalue()
    output = output[325:]
    ids = re.findall(r' {15,}(\d+)', output)
    ids = [int(i) for i in ids]

    # count the tasks across datasets
    task_counter = Counter()
    classification_ids = []
    for dataset_id in ids:
        try:
            ds = fetch_ucirepo(id=dataset_id)
            tasks = ds.metadata.get("tasks", [])
            if isinstance(tasks, str):
                tasks = [tasks]
            for task in tasks:
                task_counter[task] += 1
            if "Classification" in tasks:
                classification_ids.append(dataset_id)
        except Exception as e:
            print(f"Error with ID {dataset_id}: {e}")
    for task, count in task_counter.most_common():
        print(f"{task}: {count}")
    print("\nClassification Dataset IDs:")
    print(classification_ids)


classification_ids = [1, 2, 3, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 27, 28, 30, 31, 32, 33, 39, 40, 42, 43, 44, 45, 46, 47, 50, 52, 53, 54, 58, 59, 62, 63, 69, 70, 73, 74, 75, 76, 78, 80, 81, 82, 83, 88, 90, 91, 94, 95, 96, 101, 105, 107, 109, 110, 111, 117, 143, 144, 145, 146, 147, 148, 149, 151, 158, 159, 161, 172, 174, 176, 184, 186, 193, 198, 212, 222, 225, 229, 242, 244, 247, 257, 264, 267, 270, 277, 292, 296, 300, 312, 320, 327, 329, 332, 336, 342, 350, 352, 357, 365, 367, 372, 373, 379, 380, 383, 419, 426, 445, 451, 461, 467, 468, 471, 484, 485, 503, 519, 529, 536, 537, 545, 547, 555, 563, 565, 567, 571, 572, 579, 582, 591, 597, 601, 602, 603, 697, 713, 722, 728, 732, 755, 759, 760, 763, 799, 827, 848, 850, 856, 857, 863, 864, 878, 880, 890, 891, 911, 913, 915, 936, 938, 942, 967]
results = _create_dt_rf_comparison(classification_ids)
_create_results(results, savepath="insert your path")
_analyze_results(result_path="insert your path")
_export_class_distributions_to_excel(classification_ids, out_xlsx_path="insert your path", drop_missing_targets=True)
