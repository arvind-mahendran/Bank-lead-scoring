from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Business assumptions (example values, adjust as needed)
PROFIT_PER_SUCCESS = 100.0  # profit when a contacted lead converts
COST_PER_CONTACT = 10.0     # cost to contact a lead

