from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

DEFAULT_TARGET = "FloodProbability"
DEFAULT_ID_COLUMN = "id"

RANDOM_SEED = 42
TEST_SIZE = 0.20

MODEL_FILENAME = "lightgbm_flood_pipeline.joblib"
