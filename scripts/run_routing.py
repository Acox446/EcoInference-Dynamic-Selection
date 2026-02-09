import os
import sys
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.strategies.routing import GreenRouter
from src.config import get_active_models, get_paths

def load_models():
    models = []
    models_to_use = get_active_models()
    models_dir = get_paths()["models_dir"]
    
    print("📂 Carregant models...")
    for name in models_to_use:
        path = f"{models_dir}/{name}.pkl"
        if not os.path.exists(path):
            for f in os.listdir(models_dir):
                if f.startswith(name) and f.endswith(".pkl"):
                    path = os.path.join(models_dir, f)
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