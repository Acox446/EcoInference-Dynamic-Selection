import numpy as np
import pickle
from ..model_pool import SklearnBase

class GreenRouter:
    def __init__(self, models_list, router_path="saved_models/router.pkl"):
        self.models = models_list
        with open(router_path, "rb") as f:
            self.router = pickle.load(f)
            
        self.COSTS = {
            "Tiny": 0.0539 / 10000,   
            "Medium": 0.0550 / 10000, 
            "Large": 0.4192 / 10000,  
            "Extra": 1.4224 / 10000   
        }
        self.ROUTER_COST = 0.124840 / 12000

    def predict_sample(self, x_flat, x_img):
        decision_idx = self.router.predict(x_flat.reshape(1, -1))[0]
        
        if decision_idx >= len(self.models): decision_idx = len(self.models) - 1
            
        selected_model = self.models[decision_idx]
        
        
        if isinstance(selected_model, SklearnBase):
            x_in = x_flat.reshape(1, -1)
        else:
            x_in = x_img.reshape(1, 28, 28, 1)
            
        # Fem servir predict (no cal proba aquí, confiem en el router)
        prediction = selected_model.predict(x_in)[0]
        
        # 3. Calculem Energia: Peatge del Router + Cost del Model
        model_key = selected_model.name.split()[0]
        step_cost = self.COSTS.get(model_key, 0.0001)
        
        total_energy = self.ROUTER_COST + step_cost
        
        return prediction, total_energy, decision_idx

    def evaluate(self, X_flat, X_img, y_true):
        correct = 0
        total_energy = 0.0
        # Comptador per veure on envia les coses
        model_counts = np.zeros(len(self.models)) 
        
        print("🚀 Executant Router sobre Test Set...")
        for i in range(len(y_true)):
            pred, en, idx = self.predict_sample(X_flat[i], X_img[i])
            
            if pred == y_true[i]: correct += 1
            total_energy += en
            model_counts[idx] += 1
            
        return correct / len(y_true), total_energy / len(y_true), model_counts