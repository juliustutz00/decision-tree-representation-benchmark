import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
import pytorch_lightning as pl
import learn2learn as l2l
from .indtree_representation_utils.gentree_utils import load_data
from .indtree_representation_utils.tree_dataset_generation import trees_to_direct_encodings
from .indtree_representation_utils.metamodel import MetaModel, TreeDataModule
from .indtree_representation_utils.torch_dataset import TreeFitting
from .indtree_representation_utils.siren_multioutput import SirenSingleBranch
from .indtree_representation_utils.tree_dataset_generation import direct_encodings_to_trees
from .indtree_representation_utils.direct_encoding_utils import direct_embedding_to_repr3rows_batched

from .base import BaseRepresentation

class INDTreeRepresentation(BaseRepresentation):
    """
    Neural-network-based representation of decision trees using a common coordinate grid.
    
    Source: https://github.com/fismimosa/indtree/
    """
    def __init__(self, all_trees, X, y, encoding="direct", comparing="output", seed=0):
        """Substitutes representation function with training of a meta-model that learns to embed all collected trees into a common coordinate space."""
        # all_trees has to be a list of DecisionTreeClassifier
        self.comparing = comparing
        max_depth = np.max([tree.get_depth() for tree in all_trees]) 
        X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=seed)
        n_features_in_train = X_train.shape[1]
        n_classes_train = len(np.unique(y_train))
        n_features_in_val = X_val.shape[1]
        n_classes_val = len(np.unique(y_val))
        rf_val = RandomForestClassifier(max_depth=max_depth)
        rf_val.fit(X_val, y_val)
        n_features_in_test = X_test.shape[1]
        n_classes_test = len(np.unique(y_test))
        if encoding == "direct":
            (feats_embeddings_train, classes_embeddings_train, threshold_embeddings_train) = trees_to_direct_encodings(all_trees, max_depth, n_features_in_train, n_classes_train)
            (feats_embeddings_val, classes_embeddings_val, threshold_embeddings_val) = trees_to_direct_encodings(all_trees, max_depth, n_features_in_val, n_classes_val)
            (feats_embeddings_test, classes_embeddings_test, threshold_embeddings_test) = trees_to_direct_encodings(all_trees, max_depth, n_features_in_test, n_classes_test)
        elif encoding == "repr3rows":
            (feats_embeddings_train, classes_embeddings_train, threshold_embeddings_train) = trees_to_direct_encodings(all_trees, max_depth, n_features_in_train, n_classes_train)
            (feats_embeddings_train, classes_embeddings_train, threshold_embeddings_train) = direct_embedding_to_repr3rows_batched(feats_embeddings_train, classes_embeddings_train, threshold_embeddings_train)
            (feats_embeddings_val, classes_embeddings_val, threshold_embeddings_val) = trees_to_direct_encodings(all_trees, max_depth, n_features_in_val, n_classes_val)
            (feats_embeddings_val, classes_embeddings_val, threshold_embeddings_val) = direct_embedding_to_repr3rows_batched(feats_embeddings_val, classes_embeddings_val, threshold_embeddings_val)
            (feats_embeddings_test, classes_embeddings_test, threshold_embeddings_test) = trees_to_direct_encodings(all_trees, max_depth, n_features_in_test, n_classes_test)
            (feats_embeddings_test, classes_embeddings_test, threshold_embeddings_test) = direct_embedding_to_repr3rows_batched(feats_embeddings_test, classes_embeddings_test, threshold_embeddings_test)
        COORDS_TYPE = "binary" # can also be "integer"
        meta_dataloader_train = []
        for i in range(feats_embeddings_train.shape[0]):
            meta_dataloader_train.append(
                TreeFitting(
                    feats_embeddings_train[i],
                    threshold_embeddings_train[i],
                    classes_embeddings_train[i],
                    coords_type=COORDS_TYPE,
                    normalize=True,
                )
            )
        meta_dataloader_val = []
        for i in range(feats_embeddings_val.shape[0]):
            meta_dataloader_val.append(
                TreeFitting(
                    feats_embeddings_val[i],
                    threshold_embeddings_val[i],
                    classes_embeddings_val[i],
                    coords_type=COORDS_TYPE,
                    normalize=True,
                )
            )
        meta_dataloader_test = []
        for i in range(feats_embeddings_test.shape[0]):
            meta_dataloader_test.append(
                TreeFitting(
                    feats_embeddings_test[i],
                    threshold_embeddings_test[i],
                    classes_embeddings_test[i],
                    coords_type=COORDS_TYPE,
                    normalize=True,
                )
            )
        n_coords = meta_dataloader_train[0].coords.shape[-1]
        metamodel = l2l.algorithms.MAML
        model = SirenSingleBranch(in_features=n_coords, hidden_features=32, hidden_layers=1, out_features=n_features_in_train, out_classes=n_classes_train)
        
        self.lightning_model = MetaModel(
            model=model,
            metamodel=metamodel,
            metamodel_params=dict(
                lr=1e-3,
                first_order=False,
            ),
            optimizer_params=dict(
                lr=1e-5,
            ),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            rf_val=rf_val,
            tree_log_params=dict(
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                    ),
            weight_feat=1,
            weight_thresh=1,
            weight_class=1,
            rf_log_freq=1,
        )

        data_module = TreeDataModule(
            meta_dataloader_train,
            meta_dataloader_val,
            meta_dataloader_test,
            batch_size=1024,
        )

        trainer = pl.Trainer(
            max_epochs=10, # the authors suggest 2000 in their paper
            accelerator="auto",
            devices=1
        )
        trainer.fit(self.lightning_model, datamodule=data_module)
        self.list_train_loader = list(meta_dataloader_train)

    def represent(self, tree, X_train=None):
        pass

    def similarity(self, representation_a, representation_b):
        """Similarity can either be calculated in the embedding space directly or through the output of the meta-model."""
        coords_a, onehot_feat_a, onehot_class_a, onehot_thresh_a = self.list_train_loader[representation_a].coords.unsqueeze(0), self.list_train_loader[representation_a].onehot_feat.unsqueeze(0), self.list_train_loader[representation_a].onehot_thresh.unsqueeze(0), self.list_train_loader[representation_a].onehot_class.unsqueeze(0) 
        coords_b, onehot_feat_b, onehot_class_b, onehot_thresh_b = self.list_train_loader[representation_b].coords.unsqueeze(0), self.list_train_loader[representation_b].onehot_feat.unsqueeze(0), self.list_train_loader[representation_b].onehot_thresh.unsqueeze(0), self.list_train_loader[representation_b].onehot_class.unsqueeze(0) 
        if self.comparing == "encoding":
            coords_distance = torch.norm(coords_a - coords_b, p=2).item()
            feat_distance = torch.norm(onehot_feat_a - onehot_feat_b, p=2).item()
            thresh_distance = torch.norm(onehot_thresh_a - onehot_thresh_b, p=2).item()
            class_distance = torch.norm(onehot_class_a - onehot_class_b, p=2).item()
            total_distance = coords_distance + feat_distance + thresh_distance + class_distance
        elif self.comparing == "output":
            feat_out_a, thresh_out_a, class_out_a = self.lightning_model(coords_a, onehot_feat_a, onehot_thresh_a, onehot_class_a)
            feat_out_b, thresh_out_b, class_out_b = self.lightning_model(coords_b, onehot_feat_b, onehot_thresh_b, onehot_class_b)
            feat_distance = torch.norm(feat_out_a - feat_out_b, p=2).item()
            thresh_distance = torch.norm(thresh_out_a - thresh_out_b, p=2).item()
            class_distance = torch.norm(class_out_a - class_out_b, p=2).item()
            total_distance = feat_distance + thresh_distance + class_distance
        else:
            raise ValueError("Comparison base not implemented.")
        return 1 / (1 +  float(total_distance))
