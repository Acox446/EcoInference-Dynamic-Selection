from codecarbon import EmissionsTracker
import os

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
        os.makedirs("./logs_energy", exist_ok=True)
        self.tracker = EmissionsTracker(
            project_name=project_name, 
            output_dir="./logs_energy",
            log_level="error",  # To avoid filling the console with text
            save_to_file=True
        )
        self.energy_kwh = 0.0

    def __enter__(self):
        self.tracker.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        emissions = self.tracker.stop()
        # CodeCarbon returns emissions, but internally stores energy
        # Total energy consumed in kWh (internal function of the library)
        self.energy_kwh = self.tracker._total_energy.kWh