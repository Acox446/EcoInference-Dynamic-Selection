from codecarbon import EmissionsTracker
import os
# TODO: canviar a singleton 
class EnergyMeter:
    """
    A Context Manager to measure the energy of a code block.
    Usage:
        with EnergyMeter() as meter:
            model.predict(X)
        print(meter.energy_kwh)
    """
    def __init__(self, project_name="green_ai_replica"):
        # Store logs in a temp folder to avoid clutter
        os.makedirs("./logs", exist_ok=True)
        self.tracker = EmissionsTracker(
            project_name=project_name, 
            output_dir="./logs",
            log_level="error",  # To avoid filling the console with text
            save_to_file=True
        )
        self.energy_kwh = 0.0

    def __enter__(self):
        self.tracker.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        emissions = self.tracker.stop()
        self.energy_kwh = self.tracker._total_energy.kWh