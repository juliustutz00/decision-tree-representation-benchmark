import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold


"""
Dataset loading and preprocessing helpers for UCI and preprocessed TCGA sources.

Outputs are fold tuples:
    (X_train, X_test, y_train, y_test)
plus a `features_df` with per-feature type metadata.

Datasets searched by myself, from Bayir, Murat Ali, et al. "Topological forest." IEEE Access 10 (2022): 131711-131721. and 
Guidotti, Riccardo, et al. "Generative model for decision trees." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 19. 2024.
"""

# https://archive.ics.uci.edu/dataset/53/iris
def get_iris_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="iris", n_splits=n_splits, n_samples=n_samples, seed=seed), "iris_UCI"

# https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
def get_breast_cancer_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="breast_cancer", n_splits=n_splits, n_samples=n_samples, seed=seed), "breast_cancer_UCI"

# https://archive.ics.uci.edu/dataset/186/wine+quality
def get_wine_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="wine", n_splits=n_splits, n_samples=n_samples, seed=seed), "wine_UCI"

# https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
def get_MAGIC_gamma_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="magic_gamma", n_splits=n_splits, n_samples=n_samples, seed=seed), "MAGIC_gamma_UCI"

# https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
def get_kidney_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="kidney", n_splits=n_splits, n_samples=n_samples, seed=seed), "kidney_UCI"

# https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
def get_AI4I_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="AI4I", n_splits=n_splits, n_samples=n_samples, seed=seed), "AI4I_UCI"

# https://archive.ics.uci.edu/dataset/2/adult
def get_adult_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="adult", n_splits=n_splits, n_samples=n_samples, seed=seed), "adult_UCI"

# https://archive.ics.uci.edu/dataset/59/letter+recognition
def get_letter_recognition_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="letter_recognition", n_splits=n_splits, n_samples=n_samples, seed=seed), "letter_recognition_UCI"

# https://archive.ics.uci.edu/dataset/222/bank+marketing
def get_bank_marketing_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="bank_marketing", n_splits=n_splits, n_samples=n_samples, seed=seed), "bank_marketing_UCI"

# https://archive.ics.uci.edu/dataset/267/banknote+authentication
def get_banknote_authentication_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="banknote_authentication", n_splits=n_splits, n_samples=n_samples, seed=seed), "banknote_authentication_UCI"

# https://archive.ics.uci.edu/dataset/19/car+evaluation
def get_car_evaluation_dataset(n_splits=3, n_samples=10000, seed=0):
    return __load_ucirepo_dataset(uci_name="car_evaluation", n_splits=n_splits, n_samples=n_samples, seed=seed), "car_evaluation_UCI"

def get_aml_TCGA_dataset(n_splits=3, n_features=1000, seed=0):
    return __get_preprocessed_TCGA_dataset("aml", n_splits=n_splits, n_features=n_features, seed=seed), "aml_TCGA"

def get_bic_TCGA_dataset(n_splits=3, n_features=1000, seed=0):
    return __get_preprocessed_TCGA_dataset("breast", n_splits=n_splits, n_features=n_features, seed=seed), "bic_TCGA"

def get_coad_TCGA_dataset(n_splits=3, n_features=1000, seed=0):
    return __get_preprocessed_TCGA_dataset("colon", n_splits=n_splits, n_features=n_features, seed=seed), "coad_TCGA"

def get_gbm_TCGA_dataset(n_splits=3, n_features=1000, seed=0):
    return __get_preprocessed_TCGA_dataset("gbm", n_splits=n_splits, n_features=n_features, seed=seed), "gbm_TCGA"

def get_kirc_TCGA_dataset(n_splits=3, n_features=1000, seed=0):
    return __get_preprocessed_TCGA_dataset("kidney", n_splits=n_splits, n_features=n_features, seed=seed), "kirc_TCGA"

def get_lihc_TCGA_dataset(n_splits=3, n_features=1000, seed=0):
    return __get_preprocessed_TCGA_dataset("liver", n_splits=n_splits, n_features=n_features, seed=seed), "lihc_TCGA"

def get_lusc_TCGA_dataset(n_splits=3, n_features=1000, seed=0):
    return __get_preprocessed_TCGA_dataset("lung", n_splits=n_splits, n_features=n_features, seed=seed), "lusc_TCGA"

def get_skcm_TCGA_dataset(n_splits=3, n_features=1000, seed=0):
    return __get_preprocessed_TCGA_dataset("melanoma", n_splits=n_splits, n_features=n_features, seed=seed), "skcm_TCGA"

def get_ov_TCGA_dataset(n_splits=3, n_features=1000, seed=0):
    return __get_preprocessed_TCGA_dataset("ovarian", n_splits=n_splits, n_features=n_features, seed=seed), "ov_TCGA"

def get_sarc_TCGA_dataset(n_splits=3, n_features=1000, seed=0):
    return __get_preprocessed_TCGA_dataset("sarcoma", n_splits=n_splits, n_features=n_features, seed=seed), "sarc_TCGA"


def __get_preprocessed_TCGA_dataset(dataset_name, n_splits=3, n_features=1000, seed=0):
    """
    Load one TCGA dataset, align expression/survival tables, select top-variance
    features on each train fold, and return stratified folds.

    Source: https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html
    """
    exp = pd.read_csv(r"..\datasets\preprocessed_TCGA\\" + dataset_name + r"\exp", sep=r"\s+", header=0, index_col=0)
    exp = exp.T
    if dataset_name in ["breast", "gbm", "lung"]:
        exp.index = exp.index.str.rsplit('.', n=1).str[0]
        exp = exp[~exp.index.duplicated(keep="last")]
    exp = exp.drop_duplicates(keep="last")
    exp.index = exp.index.str.upper()
    
    survival = pd.read_csv(r"..\datasets\preprocessed_TCGA\\" + dataset_name + r"\survival", sep=r"\s+", header=0)
    survival["PatientID"] = survival["PatientID"].str.replace("-", ".", regex=False)
    survival = survival.drop_duplicates(keep="last")
    survival = survival.set_index("PatientID")
    survival = survival.drop(["Survival"], axis=1)
    survival.index = survival.index.str.upper()
    
    common_ids = exp.index.intersection(survival.index)
    exp = exp.loc[common_ids]
    survival = survival.loc[common_ids]

    mask = survival.notna().all(axis=1)
    exp = exp.loc[mask]
    survival = survival.loc[mask]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []

    for train_idx, test_idx in skf.split(exp, survival["Death"]):
        train_ids = exp.index[train_idx]
        test_ids = exp.index[test_idx]

        X_train = exp.loc[train_ids]
        X_test  = exp.loc[test_ids]
        y_train = survival.loc[train_ids]
        y_test  = survival.loc[test_ids]

        selector = VarianceThreshold()
        selector.fit(X_train)
        variances = selector.variances_
        top_1000_idx = variances.argsort()[::-1][:n_features]

        X_train = X_train.iloc[:, top_1000_idx].to_numpy()
        X_test  = X_test.iloc[:, top_1000_idx].to_numpy()

        y_train = y_train.to_numpy().ravel().astype(int)
        y_test  = y_test.to_numpy().ravel().astype(int)

        folds.append((X_train, X_test, y_train, y_test))

    features_df = pd.DataFrame({"type": ["continuous"] * n_features})

    return folds, features_df


def __load_ucirepo_dataset(uci_name, n_splits=3, n_samples=10000, seed=None):
    """
    Load one local UCI dataset folder (`X.npy`, `y.npy`, `features.csv`), enforce
    numeric feature matrix, optional subsampling, and stratified folds.

    Source: check `get_<dataset>_dataset` functions for UCI URLs.
    """
    rng = np.random.RandomState(seed)

    base_path = Path("../datasets/UCI") / uci_name

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

    if y.dtype.kind in {"U", "S", "O"}:
        y = LabelEncoder().fit_transform(y.ravel())

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

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_idx, test_idx in skf.split(X_numeric, y):
        X_train, X_test = X_numeric[train_idx], X_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        folds.append((X_train, X_test, y_train, y_test))

    return folds, features_df
