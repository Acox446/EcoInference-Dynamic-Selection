import os
import sys
import time
import pickle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import DataLoader
from src.energy import EnergyMeter
from src.model_pool import TinyModel, SmallModel, MediumModel, LargeModel, ExtraLargeModel
from src.model_pool import SklearnBase

MODELS_DIR = "saved_models"
RESULTS_FILE = "benchmark_results.csv"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print("Starting Green AI Benchmark...")
    
    loader = DataLoader()
    (X_train_flat, y_train), (X_val_flat, y_val), (X_test_flat, y_test) = loader.get_data(flatten=True)
    (X_train_img, _), (X_val_img, _), (X_test_img, _) = loader.get_data(flatten=False)

    models_list = [
        TinyModel(),
        SmallModel(),
        MediumModel(),
        LargeModel(),
        ExtraLargeModel()
    ]

    results = []

    for model in models_list:
        print(f"\n--- Processing: {model.name} ---")
        
        if isinstance(model, SklearnBase):
            X_t, X_v, X_test_curr = X_train_flat, X_val_flat, X_test_flat
        else:
            X_t, X_v, X_test_curr = X_train_img, X_val_img, X_test_img

        print("     Training...")
        start_time = time.time()
        model.fit(X_t, y_train)
        train_time = time.time() - start_time
        print(f"      Trained in {train_time:.2f}s")

        print("     Measuring Inference Energy...")
        with EnergyMeter() as meter:
            preds = model.predict(X_test_curr)
        
        accuracy = (preds == y_test).mean()
        energy_kwh = meter.energy_kwh
        # Convert to Joules for readability (1 kWh = 3.6e6 Joules)
        energy_joules = energy_kwh * 3.6e6 

        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      Energy: {energy_joules:.6f} Joules")

        results.append({
            "Model": model.name,
            "Accuracy": accuracy,
            "Energy_Joules": energy_joules,
            "Inference_Time_s": train_time # Guardem temps entrenament com ref
        })

        # F. Guardar el Model al disc (Pickle per a tots per simplificar wrapper)
        # Nota: Keras necessita un tractament especial normalment, però el pickle 
        # pot serialitzar l'objecte wrapper si no és molt complex, 
        # o podem guardar-ho manualment.
        model_path = os.path.join(MODELS_DIR, f"{model.name.split()[0]}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # 3. Mostrar i Guardar Resultats Finals
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    
    print("\n\nFINAL RANKING:")
    print(df[["Model", "Accuracy", "Energy_Joules"]].sort_values("Accuracy"))

    # 4. Quick Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Energy_Joules", y="Accuracy", s=100, hue="Model")
    plt.title("Trade-off: Accuracy vs Energy Consumption")
    plt.xlabel("Energy Consumed (Joules) - Lower is better")
    plt.ylabel("Accuracy (0-1) - Higher is better")
    plt.grid(True)
    plt.savefig("benchmark_plot.png")
    print("\nPlot saved to 'benchmark_plot.png'")

if __name__ == "__main__":
    main()