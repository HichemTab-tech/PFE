# Energy Data Processing

This project provides a minimal framework for importing and exporting data. It focuses on CSV export functionality with a simple JSON import option.

## Project Structure

```
main_program/
├── data_import.py    # Simple functions for loading data from CSV and JSON
├── data_export.py    # Basic functions for exporting data to CSV and JSON
├── main.py           # Minimal placeholder script for your own analysis
└── README.md         # This file
```

## Features

- Import data from CSV files
- Import data from JSON files
- Export data to CSV files
- Export data to JSON files

## Usage

### Basic Usage

```python
# Import the necessary functions
from data_import import load_csv_data, load_json_data
from data_export import export_csv_data, export_json_data

# Load CSV data
df = load_csv_data("data.csv")

# Export CSV data
export_csv_data(df, "exported_data.csv")

# Load JSON data
json_data = load_json_data("data.json")

# Export JSON data
export_json_data(json_data, "exported_data.json")
```

## Module Descriptions

### data_import.py

This module provides simple functions for loading data:

- `load_csv_data`: Load data from a CSV file
- `load_json_data`: Load data from a JSON file

### data_export.py

This module provides basic functions for exporting data:

- `export_csv_data`: Export DataFrame to a CSV file
- `export_json_data`: Export data to a JSON file

### main.py

This is a minimal placeholder script that demonstrates how to:

- Load data from CSV and JSON files
- Export data to CSV and JSON formats
- Add your own analysis code

## Data Format

### CSV Data

The CSV data can contain any columns and will be loaded as a pandas DataFrame.

### JSON Data

The JSON data can be any valid JSON structure. For example:

```json
{
  "Device 1": 0.0001,
  "Device 2": 0.0002
}
```

## Output

The program can export data in the following formats:
- CSV files
- JSON files

You can extend the program to implement your own analysis and custom outputs.
