import os
import sys
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.strategies.cascading import GreenCascading
from src.config import get_active_models, get_cascading_thresholds, get_paths

def load_models():
    models = []
    names = get_active_models()
    models_dir = get_paths()["models_dir"]
    
    print("Loading models...")
    for name in names:
        path = f"{models_dir}/{name}.pkl"
        if not os.path.exists(path):
            print(f"Error: Model not found at {path}")
            continue
            
        with open(path, "rb") as f:
            model = pickle.load(f)
            models.append(model)
            print(f"   Model loaded: {model.name}")
    return models

def main():
    loader = DataLoader()
    _, _, (X_test_flat, y_test) = loader.get_data(flatten=True)  # Test set flat
    _, _, (X_test_img, _) = loader.get_data(flatten=False)       # Test set img
    
    X_test_flat, X_test_img, y_test = X_test_flat, X_test_img, y_test

    models = load_models()

   
    thresholds = get_cascading_thresholds() 
    
    cascade = GreenCascading(models, thresholds)

    acc, avg_energy, counts = cascade.evaluate(X_test_flat, X_test_img, y_test)

    print("\n\n=== CASCADE RESULTS ===")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Average Energy per Sample: {avg_energy:.6f} Joules")
    print("Model Usage Distribution:")
    for i, model in enumerate(models):
        perc = (counts[i] / len(y_test)) * 100
        print(f"  - {model.name}: {counts[i]} times ({perc:.1f}%)")

if __name__ == "__main__":
    main()