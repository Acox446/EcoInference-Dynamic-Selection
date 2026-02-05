from abc import ABC, abstractmethod
import numpy as np

class GreenModel(ABC):
    """
    Model base per a models "verds" que volen ser eficients energèticament.
    """
    def __init__(self, name: str):
        self.name = name
        self.energy_consumed = 0.0 

    @abstractmethod
    def fit(self, X, y):
        """Entrena el model amb les dades d'entrenament."""
        pass

    @abstractmethod
    def predict(self, X):
        """Retorna la classe predita (un array d'enters)."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Retorna la probabilitat de cada classe. 
        Necessari per calcular la confiança (entropia) en el Cascading.
        """
        pass

    @abstractmethod
    def get_model_size(self):
        """Retorna el tamany del model en bytes (aprox)."""
        pass