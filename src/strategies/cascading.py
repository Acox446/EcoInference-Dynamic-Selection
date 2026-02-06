import numpy as np
from ..model_pool import SklearnBase, KerasBase
from ..energy import EnergyMeter

class GreenCascading:
    COSTS = {
        "Tiny": 0.0539 / 10000,    # DTree:  0.00000539 J
        "Medium": 0.0550 / 10000,  # RForest: 0.00000550 J (Empat tècnic amb Tiny!)
        "Small": 0.1260 / 10000,   # LogReg: 0.00001260 J
        "Large": 0.4192 / 10000,   # MLP:    0.00004192 J
        "Extra": 1.4224 / 10000    # CNN:    0.00014224 J
    }

    def __init__(self, models_list, thresholds):
        """
        :param models_list: Llista d'objectes model JA ENTRENATS.
            - Ex: [TinyModel(), SmallModel(), MediumModel(), LargeModel()]
            - Han de ser instàncies de SklearnBase o KerasBase.
            - L'ordre és important: del més lleuger al més pesat.
        :param thresholds: Llista de valors (0 a 1) per decidir si parem.
                           Ex: [0.9, 0.8, 0.0] -> L'últim ha de ser 0 per assegurar resposta.
        """
        self.models = models_list
        self.thresholds = thresholds
        assert len(self.models) == len(self.thresholds)

    def predict_sample(self, x_flat, x_img):
        """
        Fa la predicció per a UNA sola mostra.
        Retorna: (predicció, cost_acumulat, index_model_final)
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
        Avalua tot el dataset de test simulant la cascada.
        """
        correct = 0
        total_energy = 0.0
        model_counts = np.zeros(len(self.models)) # Per saber quin model treballa més
        
        print(f"🌊 Iniciant Cascada amb llindars: {self.thresholds}...")
        
        # Avaluem mostra a mostra (simulació realista de flux d'entrada)
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