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
    print("📂 Carregant models...")
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
                print(f"❌ ERROR: No trobo el model {name}")
                continue
        
        with open(path, "rb") as f:
            models.append(pickle.load(f))
    return models

def main():
    loader = DataLoader()
    _, (X_val_flat, y_val), _ = loader.get_data(flatten=True)
    _, (X_val_img, _), _ = loader.get_data(flatten=False)

    models = load_models()
    
    print("🔮 Generant etiquetes òptimes (Oracle)...")
    router_labels = []
    
    for i in range(len(y_val)):
        best_model_idx = -1
        
        for idx, model in enumerate(models):
            if isinstance(model, SklearnBase):
                x_in = X_val_flat[i:i+1]
            else:
                x_in = X_val_img[i:i+1]
                
            pred = model.predict(x_in)
            # Nota: Sklearn retorna array, Keras wrapper també retorna array gràcies a la nostra classe base
            if isinstance(pred, np.ndarray): pred = pred[0]
            
            if pred == y_val[i]:
                best_model_idx = idx
                break 
        
        if best_model_idx == -1:
            best_model_idx = len(models) - 1
            
        router_labels.append(best_model_idx)

    print(f"   Etiquetes generades. Distribució ideal: {np.bincount(router_labels)}")

    print("🧠 Entrenant el Router (Logistic Regression)...")
    router = LogisticRegression(max_iter=1000, class_weight='balanced') 
    router.fit(X_val_flat, router_labels)
    
    print("⚡️ Mesurant Energia d'Inferència...")
    with EnergyMeter() as meter:
        preds = router.predict(X_val_flat)
    
    accuracy = (preds == router_labels).mean()
    energy_kwh = meter.energy_kwh
    # Convertim a Joules per llegibilitat (1 kWh = 3.6e6 Joules)
    energy_joules = energy_kwh * 3.6e6 

    print(f"🔌 Energia: {energy_joules:.6f} Joules")
    print(f"✅ Router entrenat! Accuracy sobre les decisions: {accuracy:.3f}")
    
    with open("saved_models/router.pkl", "wb") as f:
        pickle.dump(router, f)
    print("💾 Guardat a saved_models/router.pkl")

if __name__ == "__main__":
    main()