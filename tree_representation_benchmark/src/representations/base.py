from abc import ABC, abstractmethod

# base class for decision tree implementations
# each representation must implement represent() and similarity()
class BaseRepresentation(ABC):
    """Abstract interface for all tree representation methods."""

    @abstractmethod
    def represent(self, tree, X_train):
        """Return representation object/vector for one decision tree."""
        pass

    @abstractmethod
    def similarity(self, representation_a, representation_b):
        """Return scalar similarity between two representations of same type."""
        pass
