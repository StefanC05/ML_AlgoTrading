# %% [markdown]
#  # Data Loading and Preparation Script
#
#  This script loads and processes historical stock data from Stooq.com in CSV files for.
#  It cleans the data, converts data types, and stores it in an HDF5 format.
#  The focus is on robust data processing and efficient memory management for large amounts of data.
#
#  The data serves as the basis for an overview of the time series characteristics,
#  and subsequent feature engineering and preprocessing  (04a_FeatureEngineering_EDA.py)
# in order to evaluate and predict the features using random forest and LSTM neural networks.

#  ##  DataRead+Merge Pipeline
#  This script implements data processing pipeline :

#  2. Collect and load all CSV files
#  3. Merge the data into a DataFrame
#  4. Clean up column names and data types
#  5. Create a multi-index structure (ticker, date)
#  6. Saving the processed data in HDF5 format


# %%
#   !pip install tables tqdm

# 0 Import Python packages
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import traceback


# %%
# 1 Loading and collecting all data files

# Constant
DATA_PATH = Path("data/raw/nyse stocks")
OUTPUT_FILE = Path("data/raw/DATA_01_assets.h5")
HDF_KEY = "assets"
START_YEAR = "2000-01-01"

# Test data Path and grab files.
try:
    # Check if the data directory exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(f": {DATA_PATH}")

    txt_files = list(DATA_PATH.rglob("*.us.txt"))

    # Testing if there are any files
    if len(txt_files) == 0:
        print(" Warning: No text files found in the  directory!")

except Exception as e:
    print(f" Errors in data collection: {e}")
    sys.exit(1)


# %%
# 2  Read single files and Merge

#         2 Read each file and collect in a list
all_dfs = []  # A list to store the individual DataFrames
empty_files = []  # A list to log empty files

try:
    # tqdm() adds a progress bar, which is helpful with many files
    for file in tqdm(txt_files, ascii="Processing files"):
        try:
            #  getsize() reads  size of all  files"
            if os.path.getsize(file) == 0:
                empty_files.append(file)  # Note the empty file
                print(f" Warning: Empty file skipped: {file}")
                continue  # Skip the rest of the loop

            df = pd.read_csv(file)

            if df.empty:
                empty_files.append(file)
                print(f"Warning: File with only headers skipped:{file}")
                continue

            # Add the DataFrame to the list
            all_dfs.append(df)

        except pd.errors.EmptyDataError as e:
            print(f" Warning: EmptyDataError while reading the file  {file}: {e}")
            empty_files.append(file)
            continue

        except pd.errors.ParserError as e:
            print(f" Error: UnicodeDecodeError while reading the file i {file}: {e}")
            continue

        except UnicodeDecodeError as e:
            print(f" Error: UnicodeDecodeError while reading the file i{file}: {e}")
            continue

        except Exception as e:
            # Catches other possible errors during reading
            print(f" Error reading the file {file}: {e}")
            print(f" Errortype: {type(e).__name__}")
            continue

    print(f"\n Successfully processed: {len(all_dfs)} files ")
    print(f" Skip: {len(empty_files)} empty files")

except Exception as e:
    print(f"Error while : {e}")
    sys.exit(1)


# %%
# 3 Combine all DataFrames into one overall DataFrame
try:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f" Merged DataFrame form: {combined_df.shape}")
    print(" Columns:", combined_df.columns.tolist())
    print("\n First 5 rows:")
    print(combined_df.head(5))

    # Convert float64 columns to float32 to reduce memory usage
    float_columns = combined_df.select_dtypes(include=["float64"]).columns
    combined_df[float_columns] = combined_df[float_columns].astype("float32")
    print(
        f"Converted {len(float_columns)} columns from float64 to float32: {list(float_columns)}"
    )

except Exception as e:
    print(f" Error combining the DataFrames: {e}")
    print(f" Fehlertyp: {type(e).__name__}")
    traceback.print_exc()
    sys.exit(1)


# %%
# 4 Cleaning of column names and data types
# So they are better to understand and readable.  "<TICKER>""  "/"1301.US"
# Column name to lowercase + no parentheses

original_columns = list(combined_df.columns)
clean_data_df = combined_df.copy()


# %%
# Clean column names      Dataset specific Cleaning
clean_data_df.columns = [col.lower() for col in original_columns]
clean_data_df.columns = clean_data_df.columns.str.replace(">", "").str.replace("<", "")

# Clean ticker name  #  suffix cleaning
clean_data_df["ticker"] = clean_data_df["ticker"].str.replace(".US", "", regex=False)
clean_data_df["date"] = pd.to_datetime(
    clean_data_df["date"].astype(str), format="%Y%m%d"
)
# Format vol column from exponential to decimal notation while preserving float32 type
if "vol" in clean_data_df.columns:
    # Convert to numeric, ensure float32 type and avoid exponential notation in display
    clean_data_df["vol"] = pd.to_numeric(clean_data_df["vol"], errors="coerce").astype(
        "float32"
    )
    # Apply formatting to avoid scientific notation when displaying
    pd.set_option(
        "display.float_format", lambda x: "%.0f" % x if x == int(x) else "%.2f" % x
    )
    print(
        "Formatted vol column to avoid exponential notation while preserving float32 type."
    )
    print(clean_data_df.sample())

    # Drop unnecessary columns, its specification for this dataset!
columns_to_drop = ["per", "time", "openint"]
existing_columns_to_drop = [
    col for col in columns_to_drop if col in clean_data_df.columns
]

if existing_columns_to_drop:
    clean_data_df = clean_data_df.drop(existing_columns_to_drop, axis=1)
    print(f"Dropped columns: {existing_columns_to_drop}")
else:
    print("Warning: No columns to drop found")


print("\n  DataFrame info after cleaning:")
clean_data_df.info()
print("\n  Sample data after cleaning:")
print(clean_data_df.head())


# %%  generate Multi-Index dataset.
# Set ticker as first level index and date as second level index
try:
    # Set multi-index with ticker as first level and date as second level
    clean_data_df = clean_data_df.set_index(["ticker", "date"])
    print("\n DataFrame with multi-index (ticker, date):")
    print(clean_data_df.head())
except Exception as e:
    print(f" Error setting multi-index structure: {e}")
    sys.exit(1)


# %%
# Save data to HDF file
try:
    clean_data_df.to_hdf(OUTPUT_FILE, key=HDF_KEY, mode="w", format="table")
    print(f"\n Data successfully saved to {OUTPUT_FILE} with key '{HDF_KEY}'")
except Exception as e:
    print(f" Error saving data to HDF5 file: {e}")
    sys.exit(1)
