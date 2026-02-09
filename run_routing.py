import pickle
import os
import numpy as np
from src.data_loader import DataLoader
from src.strategies.routing import GreenRouter

MODELS_TO_USE = ["Tiny", "Medium", "Large", "Extra"]

def load_models():
    models = []
    print("📂 Carregant models...")
    for name in MODELS_TO_USE:
        path = f"saved_models/{name}.pkl"
        if not os.path.exists(path):
            for f in os.listdir("saved_models"):
                if f.startswith(name) and f.endswith(".pkl"):
                    path = os.path.join("saved_models", f)
                    break
        with open(path, "rb") as f:
            models.append(pickle.load(f))
    return models

def main():
    loader = DataLoader()
    _, _, (X_test, y_test) = loader.get_data(flatten=True)  # Test set flat
    _, _, (X_img, _) = loader.get_data(flatten=False)       # Test set img

    models = load_models()
    router_strat = GreenRouter(models)
    
    acc, energy, counts = router_strat.evaluate(X_test, X_img, y_test)
    
    print("\n=== RESULTATS ROUTER ===")
    print(f"Precisió: {acc:.4f}")
    print(f"Energia:  {energy:.6f} J/mostra")
    print("Decisions:")
    for i, m in enumerate(models):
        print(f"  -> {m.name}: {int(counts[i])} ({counts[i]/len(y_test)*100:.1f}%)")

if __name__ == "__main__":
    main()