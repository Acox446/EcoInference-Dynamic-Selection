import numpy as np
from ..model_pool import SklearnBase, KerasBase
from ..energy import EnergyMeter
from ..config import get_energy_costs

class GreenCascading:
    COSTS = get_energy_costs()

    def __init__(self, models_list, thresholds):
        """
        :param models_list: List of ALREADY TRAINED model objects.
            - Ex: [TinyModel(), SmallModel(), MediumModel(), LargeModel()]
            - Must be instances of SklearnBase or KerasBase.
            - Order is important: from lightest to heaviest.
        :param thresholds: List of values (0 to 1) to decide if we stop.
                           Ex: [0.9, 0.8, 0.0] -> Last must be 0 to ensure response.
        """
        self.models = models_list
        self.thresholds = thresholds
        assert len(self.models) == len(self.thresholds)

    def predict_sample(self, x_flat, x_img):
        """
        Makes prediction for ONE single sample.
        Returns: (prediction, accumulated_cost, final_model_index)
        """
        total_energy = 0.0
        
        for i, model in enumerate(self.models):
            if isinstance(model, SklearnBase):
                x_input = x_flat.reshape(1, -1)
            else:
                x_input = x_img.reshape(1, 28, 28, 1)

            probs = model.predict_proba(x_input)
            
            model_key = model.name.split()[0] 
            step_cost = self.COSTS.get(model_key, 0.1) 
            total_energy += step_cost

            confidence = np.max(probs)
            prediction = np.argmax(probs)

            if confidence >= self.thresholds[i] or i == len(self.models) - 1:
                return prediction, total_energy, i, confidence
        
        return -1, 0, 0, 0

    def evaluate(self, X_flat, X_img, y_true):
        """
        Evaluates the entire test dataset simulating the cascade.
        """
        correct = 0
        total_energy = 0.0
        model_counts = np.zeros(len(self.models)) # To know which model works most
        
        print(f"🌊 Starting Cascade with thresholds: {self.thresholds}...")
        
        # Evaluate sample by sample (realistic simulation of input flow)
        n_samples = len(y_true)
        for idx in range(n_samples):
            if idx % 1000 == 0: print(f"({idx})", end="", flush=True)

            pred, energy, model_idx, conf = self.predict_sample(
                X_flat[idx], X_img[idx]
            )
            
            if pred == y_true[idx]:
                correct += 1
            
            total_energy += energy
            model_counts[model_idx] += 1

        accuracy = correct / n_samples
        avg_energy = total_energy / n_samples
        
        return accuracy, avg_energy, model_counts