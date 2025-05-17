import pandas as pd
import json
import os

def load_csv_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    df['time'] = pd.date_range('2016-01-01 05:00', periods=len(df), freq='min')
    df.set_index('time', inplace=True)
    df['month'] = df.index.strftime('%B')
    df['hour']  = df.index.hour
    return df

def load_json_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def load_devices(json_path="devices_with_w.json", csv_path="HomeC.csv"):
    """
    Loads for each device:
      - w       : cost/comfort weight
      - lambda  : comfort on/off
      - power   : max kW observed in the CSV
    """
    devices = load_json_data(json_path)
    df = load_csv_data(csv_path)

    for name, params in devices.items():
        col = next(c for c in df.columns if name in c)
        params['power'] = df[col].max()

    return devices
