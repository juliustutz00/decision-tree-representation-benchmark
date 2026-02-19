import numpy as np
import ot

from .base import BaseRepresentation

class LeafProfileRepresentation(BaseRepresentation):
    """
    Leaf-distribution profile representation compared via EMD.
    
    Source: novel representation
    """
    
    def __init__(self, criterion="l2"):
        self.criterion = criterion

    def represent(self, tree, X_train):
        """Return list of (leaf_mass, normalized_class_distribution) tuples."""
        tree = tree.tree_
        node_count = tree.node_count
        is_leaf = (tree.children_left == -1) & (tree.children_right == -1)
        
        total_samples = tree.n_node_samples[0]
        ldp = []

        for i in range(node_count):
            if is_leaf[i]:
                mass = tree.n_node_samples[i] / total_samples
                class_distribution = tree.value[i][0]
                s = class_distribution.sum()
                if s > 0:
                    class_distribution = class_distribution / s
                else:
                    class_distribution = class_distribution
                ldp.append((mass, class_distribution))
        
        return ldp
    
    def similarity(self, representation_a, representation_b):
        """Convert EMD distance to bounded similarity `1 / (1 + d)`."""
        return 1 / (1 + self.__compute_emd(representation_a, representation_b, self.criterion))

    
    def __compute_emd(self, ldp1, ldp2, criterion="l2"):
        w1, p1 = zip(*ldp1)
        w2, p2 = zip(*ldp2)

        w1 = np.array(w1)
        w2 = np.array(w2)
        p1 = np.array(p1)
        p2 = np.array(p2)

        if criterion == 'l1':
            M = np.sum(np.abs(p1[:, None, :] - p2[None, :, :]), axis=2)
        elif criterion == 'l2':
            M = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
        else:
            raise ValueError("Unsupported metric.")

        w1 = w1 / w1.sum()
        w2 = w2 / w2.sum()

        return ot.emd2(w1, w2, M)
