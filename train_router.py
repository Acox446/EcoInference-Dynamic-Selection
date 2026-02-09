import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from src.data_loader import DataLoader
from src.model_pool import SklearnBase
from src.energy import EnergyMeter

MODELS_TO_USE = ["Tiny", "Medium", "Large", "Extra"] 

def load_models():
    models = []
    print("📂 Loading models...")
    for name in MODELS_TO_USE:
        path = f"saved_models/{name}.pkl"
        if not os.path.exists(path):
            found = False
            for f in os.listdir("saved_models"):
                if f.startswith(name) and f.endswith(".pkl"):
                    path = os.path.join("saved_models", f)
                    found = True
                    break
            if not found:
                print(f"❌ ERROR: Model {name} not found")
                continue
        
        with open(path, "rb") as f:
            models.append(pickle.load(f))
    return models

def main():
    loader = DataLoader()
    _, (X_val_flat, y_val), _ = loader.get_data(flatten=True)
    _, (X_val_img, _), _ = loader.get_data(flatten=False)

    models = load_models()
    
    print("🔮 Generating optimal labels (Oracle)...")
    router_labels = []
    
    for i in range(len(y_val)):
        best_model_idx = -1
        
        for idx, model in enumerate(models):
            if isinstance(model, SklearnBase):
                x_in = X_val_flat[i:i+1]
            else:
                x_in = X_val_img[i:i+1]
                
            pred = model.predict(x_in)
            # Note: Sklearn returns array, Keras wrapper also returns array thanks to our base class
            if isinstance(pred, np.ndarray): pred = pred[0]
            
            if pred == y_val[i]:
                best_model_idx = idx
                break 
        
        if best_model_idx == -1:
            best_model_idx = len(models) - 1
            
        router_labels.append(best_model_idx)

    print(f"   Labels generated. Ideal distribution: {np.bincount(router_labels)}")

    print("🧠 Training Router (Logistic Regression)...")
    router = LogisticRegression(max_iter=1000, class_weight='balanced') 
    router.fit(X_val_flat, router_labels)
    
    print("⚡️ Measuring Inference Energy...")
    with EnergyMeter() as meter:
        preds = router.predict(X_val_flat)
    
    accuracy = (preds == router_labels).mean()
    energy_kwh = meter.energy_kwh
    # Convert to Joules for readability (1 kWh = 3.6e6 Joules)
    energy_joules = energy_kwh * 3.6e6 

    print(f"🔌 Energy: {energy_joules:.6f} Joules")
    print(f"✅ Router trained! Accuracy on decisions: {accuracy:.3f}")
    
    with open("saved_models/router.pkl", "wb") as f:
        pickle.dump(router, f)
    print("💾 Saved to saved_models/router.pkl")

if __name__ == "__main__":
    main()