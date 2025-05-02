# config.py
import os

# Base directory for datasets and outputs
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models/")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "bitcoin/")
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "notebooks/")

# Dataset-specific
CURRENCY_PAIR = 'btcusd'
DATASET_SLUG = "mczielinski/bitcoin-historical-data"
CSV_FILENAME = "btcusd_1-min_data.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

# Training parameters
WINDOW_SIZE = 60
BATCH_SIZE = 32
EPOCHS = 10
