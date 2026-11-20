"""Configuration for Water Quality Monitoring System."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Water quality parameters and safe ranges (WHO/EPA guidelines)
PARAM_RANGES = {
    "ph": (6.5, 8.5),
    "turbidity": (0, 5),       # NTU
    "dissolved_oxygen": (5, 14),  # mg/L
    "temperature": (10, 30),   # °C
    "conductivity": (0, 1500), # µS/cm
    "total_dissolved_solids": (0, 500),  # mg/L
    "nitrate": (0, 10),       # mg/L
    "fecal_coliform": (0, 0), # 0 for drinking
    "bod": (0, 3),            # mg/L
}

QUALITY_LABELS = ["Good", "Moderate", "Poor", "Unsafe"]
QUALITY_COLORS = {"Good": "#2ecc71", "Moderate": "#f1c40f", "Poor": "#e67e22", "Unsafe": "#e74c3c"}
