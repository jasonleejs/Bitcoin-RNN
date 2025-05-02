import os
import time
import torch
from datetime import datetime, timedelta, timezone

from torch.utils.data import Dataset

import pandas as pd
import requests
import warnings

from tqdm import tqdm

from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as sk_train_test_split

import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(".."))
from config import *

class BitcoinDataset(Dataset):
    
    def __init__(self, csv_path=CSV_PATH, features=None, target="Close", window_size=100, update=True):
        super().__init__()
        
        self.csv_path = csv_path
        
        if update:
            # Update CSV dataset to latest version
            update_csv_dataset(path=self.csv_path)
        
        
        # Load into a dataframe and sort by datetime
        self.df = load_bitcoin_csv(path=self.csv_path)[['Datetime', target]]
        self.df = self.df.sort_values("Datetime").reset_index(drop=True)
        
        self.target_col = target
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        
        self.scaled_data = torch.tensor(self.scaler.fit_transform(self.df[[self.target_col]]))

    
    def train_test_split(self, data=None, train_ratio=0.8, val_ratio=0.):
        if data is None:
            data = self.scaled_data

        X, y = [], []
        print(f"Creating windows of length {self.window_size}...")
        for i in tqdm(range(len(data) - self.window_size)):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size])
        X = torch.stack(X)  # shape: [num_samples, window_size, 1]
        y = torch.stack(y).squeeze()      # shape: [num_samples]

        # split into train and temp (val+test)
        train_size = train_ratio
        temp_size = 1.0 - train_size
        X_train, X_temp, y_train, y_temp = sk_train_test_split(X, y, train_size=train_size, shuffle=False)

        if val_ratio > 0:
            val_size = val_ratio / temp_size
            X_val, X_test, y_val, y_test = sk_train_test_split(X_temp, y_temp, test_size=1 - val_size, shuffle=False)
        else:
            X_val, y_val = torch.empty(0), torch.empty(0)
            X_test, y_test = X_temp, y_temp

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_scaler(self):
        return self.scaler
    
    def plot_bitcoin_timeseries(self, price_col="Close", title="BTCUSD Over Time"):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df["Datetime"], 
                self.df[price_col], 
                color="darkorange",
                linewidth=1,
                linestyle="-",
                marker=None,
                label="Close (USD)")
        plt.xlabel("Time");plt.ylabel("Price (USD)")
        plt.title(title)
        plt.grid(True);plt.legend();plt.tight_layout()
        plt.show()
# ==== OTHER FUNCTIONS ====

def load_bitcoin_csv(path):
    """
    Loads the Bitcoin dataset CSV and parses timestamps.
    
    Args:
        path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Parsed and sorted DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No dataset found at: {path}")
    
    df = pd.read_csv(path, low_memory=False)

    # Attempt to convert every column to numeric where applicable
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where Timestamp is invalid (NaN)
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # datetime version of timestamp
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s")

    return df

# Bitstamp API
def fetch_bitstamp_data(currency_pair, start_ts, end_ts, step=60, limit=1000):
    url = f"https://www.bitstamp.net/api/v2/ohlc/{currency_pair}/"
    params = {"step": step, 
              "start": int(start_ts), 
              "end": int(end_ts), 
              "limit": limit}
    
    # GET Request
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json().get("data", {}).get("ohlc", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Bitstamp data: {e}")
        return []


# Kaggle Download
def download_dataset_from_kaggle(slug, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    print("Downloading latest dataset from Kaggle...")
    
    api.dataset_download_files(slug, path=output_dir, unzip=True)


# Data Update
def check_missing_data(csv_path=CSV_PATH):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
        df = pd.read_csv(csv_path, low_memory=False)
    
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    
    last_ts = df["Timestamp"].max()
    now = datetime.now(timezone.utc) - timedelta(minutes=10)
    current_ts = int(now.timestamp())
    
    if current_ts > last_ts:
        print(f"Missing {current_ts - last_ts} seconds of data.")
        return last_ts, current_ts
    
    return None, None


def fetch_and_append_data(currency_pair, start_ts, end_ts, csv_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
        df_existing = pd.read_csv(csv_path, low_memory=False)
        df_existing["Timestamp"] = pd.to_numeric(df_existing["Timestamp"], errors="coerce")

    time_chunks = []
    chunk_size = 1000 * 60
    current_start = start_ts
    while current_start < end_ts:
        chunk_end = min(current_start + chunk_size, end_ts)
        time_chunks.append((current_start, chunk_end))
        current_start = chunk_end

    all_chunks = []
    for i, (chunk_start, chunk_end) in enumerate(time_chunks):
        print(f"Fetching chunk {i+1}/{len(time_chunks)}...")
        chunk_data = fetch_bitstamp_data(currency_pair, chunk_start, chunk_end)
        if chunk_data:
            df_chunk = pd.DataFrame(chunk_data)
            df_chunk["timestamp"] = pd.to_numeric(df_chunk["timestamp"], errors="coerce")
            df_chunk.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            all_chunks.append(df_chunk)
        time.sleep(1)

    if all_chunks:
        df_new = pd.concat(all_chunks, ignore_index=True)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all.drop_duplicates(subset="Timestamp", inplace=True)
        df_all.sort_values("Timestamp", inplace=True)
        df_all.to_csv(csv_path, index=False)
        print(f"Updated dataset saved to '{os.path.basename(csv_path)}'")
    else:
        print("No new data was fetched.")


def update_csv_dataset(path=CSV_PATH): 
    print(f"[{datetime.now(timezone.utc)}] Checking dataset...")

    # If dataset doesn't exist, download it from Kaggle
    if not os.path.exists(path):
        print("Dataset not found. Downloading full dataset...")
        
        download_dataset_from_kaggle(DATASET_SLUG, DATA_DIR)
    else:
        print("Dataset found. Checking for updates...")

    # Check for missing data
    last_ts, current_ts = check_missing_data(CSV_PATH)

    if last_ts is not None and current_ts is not None:
        fetch_and_append_data(CURRENCY_PAIR, last_ts, current_ts, CSV_PATH)
    else:
        print("Dataset is already up to date.")
    
    
if __name__ == "__main__":
    update_csv_dataset()
