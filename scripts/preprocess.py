# preprocessing stuff
# load >> explore >> clean >> filter >> validate >> save
# actually it looks like long file.. a little bit messy.. but ...putting every function in extra file ?

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualizations import (  # visualize everything.."Everyone!!!!!!!!"
    plot_ticker_lifecycle,
    plot_data_quality,
    plot_gap_analysis,
    plot_liquidity_filters,
    print_summary,
)

PERIOD_END = "2025-12-31"

# some config
input_file = "data/raw/DATA_01_assets.h5"
output_file = "data/interim/DATA_02_assets_clean.h5"
hdf_key = "assets_clean"

rolling_window = 21
threshold_pct = 0.80
gap_threshold = 5.0
min_price = 10
MIN_VOL = 250000  # kept some consistency here
min_dvol = 3500000

pd.set_option("display.max_columns", 20)


def analyze_data_quality(df):
    """Check missing values and daily ticker counts"""
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    print("\n=== MISSING VALUES ===")
    for col in df.columns:
        print(f"{col}: {missing[col]} ({missing_pct[col]}%)")

    df_temp = df.reset_index()
    daily_tickers = df_temp.groupby("date")["ticker"].nunique()

    # 21-day rolling median
    rolling_median_21d = daily_tickers.rolling(
        window=21, center=True, min_periods=10
    ).median()

    print("\n Active tickers per day")
    print(f"Median: {daily_tickers.median():.0f}")
    print(f"21-day rolling{rolling_median_21d.mean():.0f}")
    print(f"Range: {daily_tickers.min()} - {daily_tickers.max()}")

    return daily_tickers, rolling_median_21d


def analyze_inner_gaps(df_full: pd.DataFrame) -> pd.DataFrame:
    """Gap analysis within ticker lifetime."""

    closes = df_full["close"].unstack(level="ticker")
    stats = pd.DataFrame(index=closes.columns)

    stats["start"] = closes.apply(pd.Series.first_valid_index)
    stats["end"] = closes.apply(pd.Series.last_valid_index)

    def count_inner_nans(series):
        start = series.first_valid_index()
        end = series.last_valid_index()
        if start is None:
            return 0
        return series.loc[start:end].isna().sum()

    stats["inner_gaps"] = closes.apply(count_inner_nans)
    stats["total_days"] = (stats["end"] - stats["start"]).dt.days
    stats["gap_pct"] = (stats["inner_gaps"] / stats["total_days"] * 100).fillna(0)

    return stats


def create_aligned_daily_counts(
    df_calc: pd.DataFrame, column_name: str, true_label: str, false_label: str
) -> pd.DataFrame:
    counts = (
        df_calc.groupby(level="date")[column_name].value_counts().unstack(fill_value=0)
    )

    # Ensure both True and False columns exist
    for col in [True, False]:
        if col not in counts.columns:
            counts[col] = 0
    # Reindex to align dates and ensure consistent length
    counts = counts.reindex(all_dates, fill_value=0)
    return counts.rename(columns={True: true_label, False: false_label})


# Load data
try:
    df = pd.read_hdf(input_file, key="assets")
    print(f"Loaded: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(
        "Run 01_DataCollection.py first! or check the existents in main path?!"
    )


# actual start...1.start with taking data from 2000 and older. not more.
# mayby later only data after the financial crisis....
# # i will first analyse when the ticker start and how much starts and which time
# so we can see how we get the most data for a long time.
# ML model rule Nr.1: there is only one thing that beats more data..more data..better date??...
df = df[df.index.get_level_values("date") >= "2000-01-01"]
print(f"Filtered to 2000+: {df.shape}")
print(f"\nTickers count: {df.index.get_level_values('ticker').nunique()}")
# print(f"Columns: {df.columns.tolist()}")
print(
    f"Date range: {df.index.get_level_values('date').min()} -> {df.index.get_level_values('date').max()}"
)

# Ticker timespan analysis
df_temp = df.reset_index()
ticker_dates = df_temp.groupby("ticker")["date"].agg(["min", "max"])

start_years = ticker_dates["min"].dt.year.value_counts().sort_index()
end_years = ticker_dates["max"].dt.year.value_counts().sort_index()

# start year
most_frequent_start_year = start_years.idxmax()
full_data_start_year = int(most_frequent_start_year) + 1
full_data_start = f"{full_data_start_year}-01-01"
PERIOD_START = full_data_start

# max year / latest year
latest_year = end_years.index.max()
end_dates_latest = ticker_dates[ticker_dates["max"].dt.year == latest_year]["max"]
end_months = end_dates_latest.dt.month.value_counts().sort_index()

plot_ticker_lifecycle(start_years, end_years, end_months)

daily_tickers, rolling_median_21d = analyze_data_quality(df)
plot_data_quality(daily_tickers, rolling_median_21d)

# Clean: Remove ghost days (days with unusually few tickers)..didnt know that this exist..but here we go....
day_counts = df.groupby(level="date").size()
rolling_median = day_counts.rolling(
    window=rolling_window, center=True, min_periods=5
).median()

dynamic_threshold = rolling_median * threshold_pct

valid_mask = day_counts >= dynamic_threshold
valid_days = day_counts[valid_mask].index
ghost_days = day_counts[~valid_mask]

print("\n=== GHOST DAYS REMOVAL ===")
print(f"Total days: {len(day_counts)}")
print(f"days removed: {len(ghost_days)}")
df = df.loc[df.index.get_level_values("date").isin(valid_days)]
print(f"\nCleaned: {len(valid_days)} valid days")

# --------------------------------------------------------------------------------------------
# Create full index to overcome pandas "hiding..."
tickers = df.index.get_level_values("ticker").unique()
full_idx = pd.MultiIndex.from_product([tickers, valid_days], names=["ticker", "date"])

print("\nREINDEXING-----------------------------------------------------")
print(f"Sparse (length before): {len(df)} rows")
print(
    f"Full matrix: {len(full_idx):,}  = {len(tickers)} tickers x {len(valid_days)} days"
)
df_full = df.reindex(full_idx)
print(f"New NaN rows: {len(df_full) - len(df):,}")

# Gap analysis----------------------------------------------------------------------------------------------------
gap_stats = analyze_inner_gaps(df_full)

print(f"\n === INNER GAPS  ===")
print(f" ticker with gaps : {(gap_stats['inner_gaps'] > 0).sum()} / {len(gap_stats)}")
print(f"Average Gap %: {gap_stats['gap_pct'].mean():.2f}%")

# Remove tickers with >5% gaps (poor data quality)
#
bad_tickers = gap_stats[gap_stats["gap_pct"] > gap_threshold].index
df_full = df_full.drop(bad_tickers, level="ticker")

print(f" deleted: {len(bad_tickers)} Ticker (>{gap_threshold}% gaps)")

# Forward-fill small gaps (max 5 days = ~1 trading week)
df_full = df_full.groupby(level="ticker").ffill(limit=3)
print(f"Forward-Fill: gaps <=3 days filled")

plot_gap_analysis(gap_stats, gap_threshold)

# clean: check OHLC

before_drop = len(df_full)
df_clean = df_full.dropna(subset=["open", "high", "low", "close"])
dropped_rows = before_drop - len(df_clean)

print(f"\nOHLC validation")
print(f"Dropped incomplete rows: {dropped_rows}")

invalid_ohlc = (
    (df_clean["high"] < df_clean["low"])
    | (df_clean["close"] < 0)
    | (df_clean["vol"] < 0)
)

if invalid_ohlc.sum() > 0:
    print(f"Found {invalid_ohlc.sum()} invalid OHLC lines")
    df_clean = df_clean[~invalid_ohlc]
else:
    print("No OHLC issues,as if....")

print(f"Data after cleaning: {len(df_clean):,} rows")


##########################################################################
# filter: liquidity stuff
# usually we'd filter bad periods, not whole tickers, but for now removing entire tickers
print("\nLiquidity filtering")
df_calc = df_clean.copy()
df_calc["dollar_vol"] = df_calc["close"] * df_calc["vol"]
grouped = df_calc.groupby(level="ticker")

# Use olling median to identify typical liquidity levels
df_calc["med_price"] = grouped["close"].transform(
    lambda x: x.rolling(42, min_periods=42).median()
)
df_calc["med_vol"] = grouped["vol"].transform(
    lambda x: x.rolling(42, min_periods=42).median()
)
df_calc["med_dvol"] = grouped["dollar_vol"].transform(
    lambda x: x.rolling(42, min_periods=42).median()
)

ticker_stats = df_calc.groupby(level="ticker")[
    ["med_price", "med_vol", "med_dvol"]
].min()

#  boolean masks for each liquidity filter
mask_price = ticker_stats["med_price"] >= min_price
mask_vol = ticker_stats["med_vol"] >= MIN_VOL
mask_dvol = ticker_stats["med_dvol"] >= min_dvol

mask_all = mask_price & mask_vol & mask_dvol

# Extract tickers that meet all liquidity requirements
valid_tickers = ticker_stats[mask_all].index

print(f"Ticker gesamt: {len(ticker_stats)}")
print(f"Pass Price (≥${min_price}): {mask_price.sum()}")
print(f"Pass Volume (≥{MIN_VOL:,}): {mask_vol.sum()}")
print(f"Pass Dollar-Vol (≥${min_dvol:,}): {mask_dvol.sum()}")
print(
    f" Pass ALL: {len(valid_tickers)} ({len(valid_tickers) / len(ticker_stats) * 100:.1f}%)"
)

# boolean columns, track which tickers pass  which filter
# Use explicit copy to avoid Warning
df_calc = df_calc.copy()
df_calc.loc[:, "is_valid"] = df_calc.index.get_level_values("ticker").isin(
    valid_tickers
)
df_calc.loc[:, "is_price_ok"] = df_calc.index.get_level_values("ticker").isin(
    ticker_stats[mask_price].index
)
df_calc.loc[:, "is_vol_ok"] = df_calc.index.get_level_values("ticker").isin(
    ticker_stats[mask_vol].index
)
df_calc.loc[:, "is_dvol_ok"] = df_calc.index.get_level_values("ticker").isin(
    ticker_stats[mask_dvol].index
)

# Get all dates from df_calc to ensure consistent indexing
all_dates = df_calc.index.get_level_values("date").unique().sort_values()

#  valid/invalid tickers per day,WHY?: for overall liquidity visualization
daily_counts = create_aligned_daily_counts(df_calc, "is_valid", "Valid", "Invalid")


# daily county for ticker that are valid.so i can show them a graph....to much?......12.11...yES
daily_counts_price = create_aligned_daily_counts(
    df_calc, "is_price_ok", "Price_OK", "Price_Invalid"
)

daily_counts_vol = create_aligned_daily_counts(
    df_calc, "is_vol_ok", "Vol_OK", "Vol_Invalid"
)

daily_counts_dvol = create_aligned_daily_counts(
    df_calc, "is_dvol_ok", "DVol_OK", "DVol_Invalid"
)

plot_liquidity_filters(
    ticker_stats,
    daily_counts,
    daily_counts_price,
    daily_counts_vol,
    daily_counts_dvol,
    min_price,
    MIN_VOL,
    min_dvol,
)

# Filter the cleaned dataset to include only liquid tickers (AFTER plotting)
df_filtered = df_clean.loc[
    df_clean.index.get_level_values("ticker").isin(valid_tickers)
]

del df_calc

# time filter
df_reset = df_filtered.reset_index()
df_filtered = df_reset[
    (df_reset["date"] >= PERIOD_START) & (df_reset["date"] <= PERIOD_END)
].copy()
df_filtered = df_filtered.set_index(["ticker", "date"])

print(f"\nTime filter applied: {PERIOD_START} to {PERIOD_END}")
print(
    f"Remaining rows: {len(df_filtered):,} tickers: {df_filtered.index.get_level_values('ticker').nunique()}"
)

df_original = pd.read_hdf(input_file, key="assets")
print_summary(df_original, df_filtered)
print(f"\nSaving cleaned data...")
print(f"File: {output_file}")
print(f"Key: {hdf_key}")

df_filtered.to_hdf(output_file, key=hdf_key, mode="w", format="table")

# metadata ..can i use meta data to seperate / feature/target/price cols?
metadata = {
    "period_start": PERIOD_START,
    "period_end": PERIOD_END,
    "final_tickers": df_filtered.index.get_level_values("ticker").nunique(),
    "final_rows": len(df_filtered),
}

# save metadata too? maybe later
# pd.Series(metadata).to_hdf(output_file, key='metadata', mode='a')

print(f"Save successful! yeah......")
print(f" \n Final data size:")
print(f"Tickers: {metadata['final_tickers']}")
print(f"Rows: {metadata['final_rows']:,}")
print(f"Period: {PERIOD_START} to {PERIOD_END}")

print(f"\n \n Next steps:")
print(f"  >> Feature Engineering: 03_FeatureEngineering.py")
print(f"  >> Load data with: pd.read_hdf('{output_file}', key='{hdf_key}')")


def preprocess_global(
    input_file: str, output_file: str, period_start: str, period_end: str
) -> pd.DataFrame:
    df = pd.read_hdf(input_file, key="assets")
    df = df[df.index.get_level_values("date") >= "2000-01-01"]
    return df
