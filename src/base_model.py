from abc import ABC, abstractmethod
import numpy as np

class GreenModel(ABC):
    """
    Base model for "green" models that aim to be energy efficient.
    """
    def __init__(self, name: str):
        self.name = name
        self.energy_consumed = 0.0 

    @abstractmethod
    def fit(self, X, y):
        """Trains the model with training data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Returns the predicted class (an array of integers)."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Returns the probability of each class. 
        Required to calculate confidence (entropy) in Cascading.
        """
        pass

    @abstractmethod
    def get_model_size(self):
        """Returns the model size in bytes (approx)."""
        pass