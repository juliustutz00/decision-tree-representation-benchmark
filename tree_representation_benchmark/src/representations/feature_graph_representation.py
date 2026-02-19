import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import loggamma
from collections import defaultdict

from .base import BaseRepresentation

class FeatureGraphRepresentation(BaseRepresentation):
    """
    Directed feature-transition graph extracted from tree split paths.
    
    Source: https://github.com/ChristelSirocchi/urf-graphs/blob/main/utilities.r
    """

    def __init__(self, criterion="sample", X=None):
        self.criterion = criterion
        self.X = X
        feature_correlation_matrix = np.corrcoef(X, rowvar=False)
        feature_correlation_matrix = np.abs(feature_correlation_matrix)
        tmp_weight = 1 - feature_correlation_matrix
        self.weight = np.pad(tmp_weight, ((0, 1), (0, 1)), constant_values=1.0) # might make sense to choose another pad-value for the leaf nodes here

    def represent(self, tree, X_train):
        """Return weighted adjacency matrix of feature transitions (+ terminal node)."""
        feature_graph, labels = self.__compute_edge_matrix(tree, X_train)
        #self.__visualize_feature_graph(feature_graph, labels) # enable if you want to visualize the feature graph
        return feature_graph
    
    def similarity(self, representation_a, representation_b):
        """Similarity as inverse correlation-adjusted Frobenius distance."""
        #return nx.graph_edit_distance(nx.from_numpy_array(representation_a), nx.from_numpy_array(representation_b)) #GED
        #return float(cosine_similarity(representation_a.flatten().reshape(1, -1), representation_b.flatten().reshape(1, -1))) #cosine similarity
        #return np.sum((representation_a > 0).astype(int) & (representation_b > 0).astype(int)) / np.sum((representation_a > 0).astype(int) | (representation_b > 0).astype(int)) if np.sum((representation_a > 0).astype(int) | (representation_b > 0).astype(int)) != 0 else 0.0 #Jaccard
        #return self.NMI(representation_a.shape[0], self.edge_matrix_to_edge_set(representation_a), self.edge_matrix_to_edge_set(representation_b)) #https://www.nature.com/articles/s42005-024-01830-3 (normalized mutual information)
        #return self.DCNMI(representation_a.shape[0], self.edge_matrix_to_edge_set(representation_a), self.edge_matrix_to_edge_set(representation_b)) #https://www.nature.com/articles/s42005-024-01830-3 (degree-corrected normalized mutual information)
        #return np.linalg.norm((representation_a - representation_b), "fro") * -1 # Frobenius norm
        return 1 / (1 + self.compute_correlated_frobenius_norm(representation_a, representation_b)) # Correlation-adjusted Frobenius norm
    
        
    def __fixation_traverse_tree(self, tree, node_id, X, edge_matrix, sample_size, feature_names, level=1, feature_counts={}):
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        if children_left[node_id] == children_right[node_id]:
            return edge_matrix
        split_feature_idx = feature[node_id]
        split_value = threshold[node_id]
        left_child = children_left[node_id]
        right_child = children_right[node_id]

        left_samples = X[:, split_feature_idx] <= split_value
        right_samples = X[:, split_feature_idx] > split_value

        parent_name = feature_names[split_feature_idx]
        left_name = "T" if feature[left_child] == -2 else feature_names[feature[left_child]]
        right_name = "T" if feature[right_child] == -2 else feature_names[feature[right_child]]

        if self.criterion == "present":
            left_weight = 1.0
            right_weight = 1.0
        elif self.criterion == "fixation":
            feature_idx = feature[node_id]
            feature_counts[feature_idx] = feature_counts.get(feature_idx, 0) + 1
            left_weight = feature_counts[feature_idx]
            right_weight = feature_counts[feature_idx]
        elif self.criterion == "level":
            left_weight = 1.0 / level
            right_weight = 1.0 / level
        elif self.criterion == "sample":
            left_weight = np.sum(left_samples) / sample_size
            right_weight = np.sum(right_samples) / sample_size
        else:
            raise ValueError("Undefined criterion.")

        i_parent = feature_names.index(parent_name)
        i_left = len(feature_names) if left_name == "T" else feature_names.index(left_name)
        i_right = len(feature_names) if right_name == "T" else feature_names.index(right_name)

        edge_matrix[i_parent, i_left] += left_weight
        edge_matrix[i_parent, i_right] += right_weight

        if feature[left_child] != -2:
            edge_matrix = self.__fixation_traverse_tree(tree, left_child, X[left_samples], edge_matrix, sample_size, feature_names, level=level + 1, feature_counts=feature_counts)
        if feature[right_child] != -2:
            edge_matrix = self.__fixation_traverse_tree(tree, right_child, X[right_samples], edge_matrix, sample_size, feature_names, level=level + 1, feature_counts=feature_counts)

        return edge_matrix
    
    def __compute_edge_matrix(self, tree, X_train):
        n_features = X_train.shape[1]
        feature_names = [f"X{i}" for i in range(n_features)]
        edge_matrix = np.zeros((n_features + 1, n_features + 1), dtype=float)
        edge_matrix = self.__fixation_traverse_tree(tree.tree_, 0, X_train, edge_matrix, X_train.shape[0], feature_names)
        return edge_matrix, feature_names + ["T"]
    
    def __visualize_feature_graph(self, edge_matrix, labels, weight_threshold=0.0):
        G = nx.DiGraph()
        for i, src in enumerate(labels):
            for j, dst in enumerate(labels):
                weight = edge_matrix[i, j]
                if weight > weight_threshold:
                    G.add_edge(src, dst, weight=weight)

        pos = nx.spring_layout(G, seed=42)
        edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]

        nx.draw(
            G, pos,
            with_labels=True,
            node_size=1500,
            node_color="lightblue",
            arrowsize=20,
            width=edge_weights,
            font_size=10
        )
        plt.title("Feature Graph (type = 'sample')")
        plt.show()


    def logmultiset(self, N,K):
        """logarithm of multiset coefficient"""
        return loggamma(N+K-1+1) - loggamma(K+1) - loggamma(N-1+1)

    def zero_log(self, x):
        """log of zero is zero"""
        if x <= 0: return 0
        else: return np.log(x)

    def ent(self, vec):
        """entropy of a distribution"""
        vec  = np.array(vec)/sum(vec)
        return -sum([x*self.zero_log(x) for x in vec])

    def jaccard(self, A, B):
        """Jaccard index of sets A and B"""
        return len(A & B) / (len(A) + len(B) - len(A & B))
    
    def NMI(self, N,e1,e2):
        """normalized mutual information between N-node graphs with edge sets e1, e2"""
        Nc2 = N*(N-1)/2
        E1,E2,E12,Union = len(e1),len(e2),len(e1.intersection(e2)),len(e1.union(e2))
        p1,p2,p12 = E1/Nc2,E2/Nc2,E12/Nc2
        H1,H2 = self.ent([p1,1-p1]), self.ent([p2,1-p2])
        MI = H1 + H2 - self.ent([p12,p1-p12,p2-p12,1-p1-p2+p12])
        NMI = (2*MI+1e-100)/(H1+H2+1e-100) # negligibly small constants for the empty and complete graphs
        return NMI

    def DCNMI(self, N,e1,e2):
        """degree-corrected normalized mutual information between N-node graphs with edge sets e1, e2"""
        adj1,adj2 = defaultdict(set),defaultdict(set)
        for e in e1:
            i,j = e
            if not(i in adj1): adj1[i] = set([])
            if not(j in adj1): adj1[j] = set([])
            adj1[i].add(j)
            adj1[j].add(i)
        for e in e2:
            i,j = e
            if not(i in adj2): adj2[i] = set([])
            if not(j in adj2): adj2[j] = set([])
            adj2[i].add(j)
            adj2[j].add(i)
        DCH1,DCH2,DCMI = 0,0,0
        for i in range(N):
            p1i,p2i,p12i = len(adj1[i])/N,len(adj2[i])/N,len(adj1[i].intersection(adj2[i]))/N 
            DCH1 += self.ent([p1i,1-p1i])
            DCH2 += self.ent([p2i,1-p2i])
            DCMI += self.ent([p1i,1-p1i]) + self.ent([p2i,1-p2i]) - self.ent([p12i,p1i-p12i,p2i-p12i,1-p1i-p2i+p12i])
        DCNMI = (2*DCMI+1e-100)/(DCH1+DCH2+1e-100) # negligibly small constants for the empty and complete graphs
        return DCNMI

    def mesoNMI(self, N,e1,e2,partition):
        """mesoscale normalized mutual information between N-node graphs with edge sets e1, e2 and reference partition"""
        B = len(set(partition))
        Bc2 = B*(B-1)/2
        E1,E2 = len(e1),len(e2)
        
        table1,table2 = defaultdict(int),defaultdict(int)
        for e in e1:
            i,j = e
            r,s = sorted([partition[i],partition[j]])
            if not((r,s) in table1): table1[(r,s)] = 0
            table1[(r,s)] += 1
        for e in e2:
            i,j = e
            r,s = sorted([partition[i],partition[j]])
            if not((r,s) in table2): table2[(r,s)] = 0
            table2[(r,s)] += 1
        
        E12 = 0
        pairs = set(list(table1.keys())+list(table2.keys()))
        for pair in pairs:
            E12 += min(table1[pair],table2[pair])
            
        H1,H2,H12 = self.logmultiset(Bc2+B,E1),self.logmultiset(Bc2+B,E2),self.logmultiset(Bc2+B,E1+E2-E12)
        I = H1 + H2 - H12
        I0 = H1 + H2 - self.logmultiset(Bc2+B,E1+E2)

        return (I - I0 +1e-100)/((H1+H2)/2 - I0 +1e-100)

    def edge_matrix_to_edge_set(self, edge_matrix, threshold=0.0):
        edges = set()
        N = edge_matrix.shape[0]
        for i in range(N):
            for j in range(N):
                if edge_matrix[i, j] > threshold:
                    edges.add((i, j))
        return edges
    
    def compute_correlated_frobenius_norm(self, matrix_1, matrix_2):
        if matrix_1.shape != matrix_2.shape:
            raise ValueError("2 Graphs have to have the same dimensions in order to be compared.")
        
        diff = (matrix_1 - matrix_2)**2 * self.weight

        return np.sqrt(np.sum(diff))
