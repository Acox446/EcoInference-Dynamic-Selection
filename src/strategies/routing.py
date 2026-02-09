import numpy as np
import pickle
from ..model_pool import SklearnBase
from ..config import get_energy_costs, get_models_config, get_paths

class GreenRouter:
    def __init__(self, models_list, router_path=None):
        self.models = models_list
        
        if router_path is None:
            router_path = get_paths()["router_path"]
        with open(router_path, "rb") as f:
            self.router = pickle.load(f)
            
        self.COSTS = get_energy_costs()
        self.ROUTER_COST = get_models_config()["router"]["cost"]

    def predict_sample(self, x_flat, x_img):
        decision_idx = self.router.predict(x_flat.reshape(1, -1))[0]
        
        if decision_idx >= len(self.models): decision_idx = len(self.models) - 1
            
        selected_model = self.models[decision_idx]
        
        
        if isinstance(selected_model, SklearnBase):
            x_in = x_flat.reshape(1, -1)
        else:
            x_in = x_img.reshape(1, 28, 28, 1)
            
        # Use predict (no need for proba here, we trust the router)
        prediction = selected_model.predict(x_in)[0]
        
        # 3. Calculate Energy: Router toll + Model cost
        model_key = selected_model.name.split()[0]
        step_cost = self.COSTS.get(model_key, 0.0001)
        
        total_energy = self.ROUTER_COST + step_cost
        
        return prediction, total_energy, decision_idx

    def evaluate(self, X_flat, X_img, y_true):
        correct = 0
        total_energy = 0.0
        # Counter to see where samples are routed
        model_counts = np.zeros(len(self.models)) 
        
        print("🚀 Running Router on Test Set...")
        for i in range(len(y_true)):
            pred, en, idx = self.predict_sample(X_flat[i], X_img[i])
            
            if pred == y_true[i]: correct += 1
            total_energy += en
            model_counts[idx] += 1
            
        return correct / len(y_true), total_energy / len(y_true), model_counts