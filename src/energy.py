from codecarbon import EmissionsTracker
import os

class EnergyMeter:
    """
    Un Context Manager per mesurar l'energia d'un bloc de codi.
    Ús:
        with EnergyMeter() as meter:
            model.predict(X)
        print(meter.energy_kwh)
    """
    def __init__(self, project_name="green_ai_replica"):
        # Guardem els logs en una carpeta temp per no molestar
        os.makedirs("./logs_energy", exist_ok=True)
        self.tracker = EmissionsTracker(
            project_name=project_name, 
            output_dir="./logs_energy",
            log_level="error",  # Perquè no ompli la consola de text
            save_to_file=True
        )
        self.energy_kwh = 0.0

    def __enter__(self):
        self.tracker.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        emissions = self.tracker.stop()
        # CodeCarbon retorna emissions, però internament guarda l'energia
        # L'energia total consumida en kWh (funció interna de la llibreria)
        self.energy_kwh = self.tracker._total_energy.kWh