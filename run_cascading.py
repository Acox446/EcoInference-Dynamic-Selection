import pickle
import os
import numpy as np
from src.data_loader import DataLoader
from src.strategies.cascading import GreenCascading

# Carregar models guardats
def load_models():
    models = []
    names = ["Tiny", "Small", "Medium", "Large", "Extra"] 
    
    print("📂 Carregant models...")
    for name in names:
        path = f"saved_models/{name}.pkl"
        if not os.path.exists(path):
            print(f"❌ Error: No trobo {path}")
            continue
            
        with open(path, "rb") as f:
            model = pickle.load(f)
            models.append(model)
            print(f"   Model carregat: {model.name}")
    return models

def main():
    loader = DataLoader()
    (X_test_flat, y_test), _, _ = loader.get_data(flatten=True) # Test set flat
    _, _, (X_test_img, _) = loader.get_data(flatten=False)      # Test set img
    # Només farem servir 500 mostres per no esperar hores (CodeCarbon és lent arrencant/parant)
    limit = 500 
    X_test_flat, X_test_img, y_test = X_test_flat[:limit], X_test_img[:limit], y_test[:limit]

    models = load_models()

    # 2. Definir Estratègia (Llindars)
    # Provem una configuració agressiva
    # Si l'Arbre té 80% seguretat -> Acceptem.
    # Si RF té 70% seguretat -> Acceptem.
    # ...
    thresholds = [0.8, 0.7, 0.7, 0.6, 0.0] 
    
    cascade = GreenCascading(models, thresholds)

    # 3. Executar
    acc, avg_energy, counts = cascade.evaluate(X_test_flat, X_test_img, y_test)

    # 4. Resultats
    print("\n\n=== RESULTATS CASCADA ===")
    print(f"Precisió Global: {acc:.4f}")
    print(f"Energia Mitjana/Mostra: {avg_energy:.6f} Joules")
    print("Distribució d'ús de models:")
    for i, model in enumerate(models):
        perc = (counts[i] / limit) * 100
        print(f"  - {model.name}: {counts[i]} vegades ({perc:.1f}%)")

if __name__ == "__main__":
    main()