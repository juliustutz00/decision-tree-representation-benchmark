# Decision Tree Representation Benchmark

Comparing decision trees is inherently non-trivial due to their structure. While there exist potential tree distance measures, these focus on structure do not capture the functionality of the underlying decision trees.

Representations of said decision trees enable a structural and functional comparison by abstracting some information and thereby gaining the ability to quantify similarity.

This benchmark therefore explores the usefulness of different decision tree representations by 
(i) assessing the representations in an isolated setting by using controlled perturbations and measuring correlations between representation distances, performance differences, and structural distances, and
(ii) measuring the representation’s effectiveness on downstream tasks by using their distances for a diverse subforest selection which is then compared against subforests chosen at random or solely based on out-of-bag (OOB) accuracy.

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

## Subforest Selection

## Perturbations
