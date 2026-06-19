import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer

DATA_ROOT = Path(os.environ["SUBFOREST_WORKDIR"]).resolve() / "datasets"

if not DATA_ROOT.exists():
    raise RuntimeError(f"DATA_ROOT does not exist: {DATA_ROOT}")

# datasets searched by myself, from 
# Bayir, Murat Ali, et al. "Topological forest." IEEE Access 10 (2022): 131711-131721.
# and 
# Guidotti, Riccardo, et al. "Generative model for decision trees." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 19. 2024.

# UCI
# https://archive.ics.uci.edu/dataset/53/iris
def get_iris_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="iris", n_splits=n_splits, n_samples=n_samples, seed=seed), "iris_UCI"

def get_cervical_cancer_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="cervical_cancer", n_splits=n_splits, n_samples=n_samples, seed=seed), "cervical_cancer_UCI"

def get_connectionist_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="connectionist", n_splits=n_splits, n_samples=n_samples, seed=seed), "connectionist_UCI"

def get_credit_approval_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="credit_approval", n_splits=n_splits, n_samples=n_samples, seed=seed), "credit_approval_UCI"

def get_cylinder_bands_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="cylinder_bands", n_splits=n_splits, n_samples=n_samples, seed=seed), "cylinder_bands_UCI"

def get_DARWIN_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="DARWIN", n_splits=n_splits, n_samples=n_samples, seed=seed), "DARWIN_UCI"

def get_heart_disease_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="heart_disease", n_splits=n_splits, n_samples=n_samples, seed=seed), "heart_disease_UCI"

def get_japanese_credit_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="japanese_credit", n_splits=n_splits, n_samples=n_samples, seed=seed), "japanese_credit_UCI"

def get_musk_1_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="musk_1", n_splits=n_splits, n_samples=n_samples, seed=seed), "musk_1_UCI"

def get_statlog_heart_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="statlog_heart", n_splits=n_splits, n_samples=n_samples, seed=seed), "statlog_heart_UCI"

def get_waveform_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="waveform", n_splits=n_splits, n_samples=n_samples, seed=seed), "waveform_UCI"


def __load_ucirepo_dataset(uci_name, n_splits=3, n_samples=10000, seed=None):
    rng = np.random.RandomState(seed)

    base_path = DATA_ROOT / "UCI" / uci_name

    X = np.load(base_path / "X.npy", allow_pickle=True)
    y = np.load(base_path / "y.npy", allow_pickle=True)
    vars_df = pd.read_csv(base_path / "features.csv")

    features_df = vars_df[vars_df["role"] == "Feature"][["type"]].reset_index(drop=True)

    X = np.asarray(X)

    # drop categorical and binary features
    allowed_types = {"continuous", "integer", "int", "numeric", "real"}
    feature_types = features_df["type"].astype(str).str.lower()
    keep_mask = feature_types.isin(allowed_types)
    keep_indices = np.where(keep_mask)[0]

    X = X[:, keep_indices]
    features_df = features_df.iloc[keep_indices].reset_index(drop=True)

    y = np.asarray(y)

    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.ravel()
        elif y.shape[0] == X.shape[0]:
            y = y[:, 0]
        elif y.shape[1] == X.shape[0]:
            y = y.T[:, 0]
        else:
            y = y.ravel()

    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]

    if y.dtype.kind in {"U", "S", "O"} or uci_name == "statlog_heart":
        y = LabelEncoder().fit_transform(y.ravel())
    y = y.astype(np.int64)

    if X.shape[0] != y.shape[0]:
        min_n = min(X.shape[0], y.shape[0])
        X = X[:min_n]
        y = y[:min_n]

    if n_samples is not None and n_samples < len(X):
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    X_obj = np.array(X, dtype=object)
    n_features_in = X_obj.shape[1]

    for col_idx in range(n_features_in):
        try:
            ftype = str(features_df.loc[col_idx, "type"]).lower()
        except Exception:
            ftype = "continuous"

        col = X_obj[:, col_idx]

        if ftype in ["categorical", "binary"]:
            le = LabelEncoder()
            X_obj[:, col_idx] = le.fit_transform(col.astype(str))
        else:
            try:
                X_obj[:, col_idx] = col.astype(float)
            except Exception:
                coerced = pd.to_numeric(col, errors="coerce")
                mean = np.nanmean(coerced)
                if np.isnan(mean):
                    mean = 0.0
                X_obj[:, col_idx] = np.where(np.isnan(coerced), mean, coerced)

    X_numeric = X_obj.astype(float)
    if np.isnan(X_numeric).any():
        imputer = KNNImputer(n_neighbors=5)
        X_numeric = imputer.fit_transform(X_numeric)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_idx, test_idx in skf.split(X_numeric, y):
        X_train, X_test = X_numeric[train_idx], X_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        folds.append((X_train, X_test, y_train, y_test))

    return folds, features_df



def _random_resample(X, y, mode, random_state):
    # mode: "under" | "over"
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)

    if len(classes) < 2:
        return X, y

    target = counts.min() if mode == "under" else counts.max()

    sampled_idx = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.flatnonzero(y == cls)
        if mode == "under":
            if cnt > target:
                cls_idx = rng.choice(cls_idx, size=target, replace=False)
        else:  # over
            if cnt < target:
                cls_idx = rng.choice(cls_idx, size=target, replace=True)
        sampled_idx.append(cls_idx)

    sampled_idx = np.concatenate(sampled_idx)
    rng.shuffle(sampled_idx)

    return X[sampled_idx], y[sampled_idx]

    np.save("F:/SUBFOREST/datasets/UCI/" + str(name) + "/X.npy", X.to_numpy())
    np.save("F:/SUBFOREST/datasets/UCI/" + str(name) + "/y.npy", y.to_numpy())
    features_df.to_csv("F:/SUBFOREST/datasets/UCI/" + str(name) + "/features.csv", index=False)'''
