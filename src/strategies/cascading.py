import numpy as np
from ..model_pool import SklearnBase, KerasBase
from ..energy import EnergyMeter

class GreenCascading:
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

            with EnergyMeter() as meter:
                probs = model.predict_proba(x_input)
            
            # Sumem el cost d'aquest pas
            # NOTA: CodeCarbon té un overhead alt per mesures tan petites (ms).
            # En una simulació real, sumariem el cost mitjà que hem trobat al benchmark.
            # Però aquí farem servir la lectura real (compte, serà sorollosa).
            total_energy += meter.energy_kwh * 3.6e6 # a Joules

            # 3. Mirem la confiança (Max probability)
            confidence = np.max(probs)
            prediction = np.argmax(probs)

            # 4. Decidim: Parem o seguim?
            # Si la confiança supera el llindar O és l'últim model
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
            # Progres (opcional)
            if idx % 1000 == 0: print(f".", end="", flush=True)

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