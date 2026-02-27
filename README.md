# Decision Tree Representation Benchmark

Comparing decision trees is inherently non-trivial due to their structure. While there exist potential tree distance measures, these focus on structure do not capture the functionality of the underlying decision trees.

Representations of said decision trees enable a structural and functional comparison by abstracting some information and thereby gaining the ability to quantify similarity.

This benchmark therefore explores the usefulness of different decision tree representations by  
(i) assessing the representations in an isolated setting by using controlled perturbations and measuring correlations between representation distances, performance differences, and structural distances, and  
(ii) measuring the representation’s effectiveness on downstream tasks by using their distances for a diverse subforest selection which is then compared against subforests chosen at random or solely based on out-of-bag (OOB) accuracy.

## Quickstart
```sh
sudo apt install git-lfs
git clone https://github.com/juliustutz00/decision-tree-representation-benchmark.git
cd decision-tree-representation-benchmark

# THE FOLLOWING 5 STEPS ARE RECOMMENDED BUT NOT NECESSARY
curl https://pyenv.run | bash
pyenv install 3.10.19 # or 3.12.9
python -m venv .venv
source .venv/bin/activate
pyenv local 3.10.19 # or 3.12.9

git lfs install
git lfs pull
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m src.benchmark_runner
```

The default run configuration is in [`src/benchmark_runs.yaml`](tree_representation_benchmark/src/benchmark_runs.yaml) and is loaded by [`main`](tree_representation_benchmark/src/benchmark_runner.py) in [`src/benchmark_runner.py`](tree_representation_benchmark/src/benchmark_runner.py).
If you run the code on Windows, you'll need to add "results_root: ..\results" to the default of [`src/benchmark_runs.yaml`](tree_representation_benchmark/src/benchmark_runs.yaml).

### Note
There are multiple parameters that heavily influence runtime; adjust those accordingly to your needs
- [`src/benchmark_utils.py`](tree_representation_benchmark/src/benchmark_utils.py) (line 26): max_depth=10 (higher depths can easily lead to very long runtimes and huge memory requirements (because INDTree's performance and memory usage scales with the max_depth of any tree it tries to learn))
- [`src/representations/indtree_representation.py`](tree_representation_benchmark/src/representations/indtree_representation.py) (line 115): batch_size=1024 (adjust based on your hardware)
- [`src/representations/indtree_representation.py`](tree_representation_benchmark/src/representations/indtree_representation.py) (line 119): max_epochs=10 (the authors suggest 2000; for this task, 10 might be enough)
- [`src/benchmark_runs.yaml`](tree_representation_benchmark/src/benchmark_runs.yaml) (lines 5-6): representation_benchmark:true AND subforest_selection:true (only choose what you need)
- [`src/benchmark_runs.yaml`](tree_representation_benchmark/src/benchmark_runs.yaml) (line 10): n_splits:3 (number of cross-validation folds multiplies time needed)
- [`src/benchmark_runs.yaml`](tree_representation_benchmark/src/benchmark_runs.yaml) (line 27): random_forest_size:300 (increasing number of decision trees in random forest drastically increases runtime and memory requirements)
- [`src/benchmark_runs.yaml`](tree_representation_benchmark/src/benchmark_runs.yaml) (lines 25-26): intensities:[0.2,0.4,0.6,0.8,1] AND perturbation_runs:1 (perturbation experiment has to deal with random_forest_size \* |intensities| \* |perturbations| \* |perturbation_runs| trees)

## Data expectations

The loaders in [`src/data_utils.py`](tree_representation_benchmark/src/data_utils.py) (e.g. [`__load_ucirepo_dataset`](tree_representation_benchmark/src/data_utils.py), [`__get_preprocessed_TCGA_dataset`](tree_representation_benchmark/src/data_utils.py)) expect downloaded datasets.

Expected folder layout (per UCI dataset):
```
datasets/
  UCI/
    <uci_name>/
      X.npy
      y.npy
      features.csv
```

Expected folder layout (per TCGA dataset):
```
datasets/
  preprocessed_TCGA/
    <name>/
      exp
      survival
```

Notes: 
* Categorical/binary features are **dropped** as sklearn DecisionTreeClassifier cannot use them (only continuous/integer-like types are kept).
* Splits are created via stratified cross-validation.
* If you pull `datasets/` as shown in the quickstart, the code can run “as-is”.

 ## Representations

| **Representation** | **Reference** | **Type** | **Distance** |
| --- | --- | --- | --- |
| Tree Descriptor | (novel) | Metric Vector | Cosine Distance |
| Leaf Profile | (novel) | Distribution Vector | Earth Mover's Distance |
| Feature Graph | [Sirocchi et al.](https://doi.org/10.1186/s13040-025-00430-3) | Graph | Correlation-adjusted Frobenius Distance |
| Topological Forest | [Bayir et al.](https://doi.org/10.1109/ACCESS.2022.3229008) | Metric Vector | Mapper Graph Shortest-Path |
| INDTree | [Spinnato et al.](https://www.esann.org/sites/default/files/proceedings/2025/ES2025-85.pdf) | Function | Embedding Space Euclidean Distance |

The following figure shows how a simple decision tree is converted into each respective representation.
<img width="1158" height="1098" alt="image" src="https://github.com/user-attachments/assets/cd3ac300-680c-4bb1-b182-99c08bf1675d" />

## Representation Benchmark

This benchmark evaluates how well different tree representations capture meaningful differences between decision trees. For each base tree, we compute a representation embedding and compare it to embeddings of systematically perturbed versions of the same tree. The resulting representation distances are then related to (i) changes in predictive performance and (ii) a structural distance based on tree edit distance, to assess whether a representation is sensitive to relevant model changes.

## Subforest Selection

Subforest selection studies how to choose a small, diverse subset of trees from a larger random forest while retaining predictive quality. The code builds pairwise distance matrices from the selected representations and applies clustering-based selection (e.g., k-medoids, agglomerative, density-based) to pick representative trees. Selected subforests are evaluated on held-out test data and compared against baselines such as random selection and top-OOB trees.

## Perturbations

Perturbations are controlled modifications applied directly to sklearn DecisionTreeClassifiers to generate tree variants with different degrees of change. Implemented perturbations include threshold changes, feature changes, node swaps, node removals, node additions, and combined perturbations. After each perturbation, affected subtree statistics are updated so the modified tree remains executable, enabling consistent measurement of both representation distances and structural differences.

## Structural difference (tree edit distance)

Structural difference is computed via Zhang–Shasha tree edit distance using `zss` in
[`src/structural_difference.py`](tree_representation_benchmark/src/structural_difference.py) ([`compute_structural_difference`](tree_representation_benchmark/src/structural_difference.py)).

### Node labels
Each sklearn node is mapped to a `zss.Node` label:
* Leaf: `"Leaf"`
* Internal: `"f<feature_idx>:<threshold>"` (threshold formatted to 3 decimals)

### Implemented costs
The label distance is defined in `substitution_cost` inside [`compute_structural_difference`](tree_representation_benchmark/src/structural_difference.py):

* **Insertion / deletion**: if either label is empty (`''`), cost = `1`
* **Exact match**: cost = `0`
* **Leaf vs internal**: cost = `1`
* **Internal vs internal, different feature**: cost = `2`
* **Internal vs internal, same feature**: cost is normalized threshold difference:
  Let $r = \max(X[:, f]) - \min(X[:, f])$ on the training data and
  $d = \frac{|t_a - t_b|}{r}$ (or $0$ if $r=0$). The cost is `min(d, 0.5)`.

## Outputs

### Representation Benchmark
* [`metadata.jsonl`](results/perturbation_results/metadata.jsonl): lists all parameters used for the experiment
* [`performance_ted_correlations.csv`](results/perturbation_results/performance_ted_correlations.csv): lists correlations of representation similarity, accuracy, and tree edit distance for all representations
* [`perturbations.csv`](results/perturbation_results/perturbations.csv): lists representation similarity, performance, and tree edit distance for all perturbed trees in comparison to the original unperturbed tree
* [`representation_correlations.csv`](results/perturbation_results/representation_correlations.csv): lists the correlations between the representation similarities of the perturbed trees across all representations (e.g. how similar any two representations behave when calculating decision tree distances)
* [`{representation}_similarity_performance.png`](results/plots/{representation}_similarity_performance.png): plots the correlation between representation similarity and accuracy for each respective representation for all perturbed trees (visual representation of performance_ted_correlations.csv)
* [`{representation}_similarity_structural_distance.png`](results/plots/{representation}_similarity_structural_distance.png): plots the correlation between representation similarity and tree edit distance for each respective representation for all perturbed trees (visual representation of performance_ted_correlations.csv)

### Subforest Selection
* [`metadata.jsonl`](results/subforest_results/metadata.jsonl): lists all parameters used for the experiment
* [`subforest.csv`](results/subforest_results/subforest.csv): lists different metrics of representation-derived subforests; done for multiple combinations of subforest sizes and selection methods; takes full random forest, randomly chosen trees, and trees with top OOB-accuracy as baselines

## Limitations

* Only works for classification tasks; regression tasks are not supported
* Only works with numerical features; categorical/binary features are not supported (except for the perturbations, they support categorical data)
