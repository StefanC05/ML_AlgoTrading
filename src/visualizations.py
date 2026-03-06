import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuration
plt.style.use("seaborn-v0_8-darkgrid")


def plot_ticker_lifecycle(
    start_years: pd.Series, end_years: pd.Series, end_months: pd.Series
) -> None:
    """Visualize start/end date distribution of tickers (yearly bars only)
    so we can see in wich year most of the ticker starts.

    Target: get most data from all the files. by choosing the median start year.
    """

    # Create plot layout with 2 subplots for yearly date distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot start date distribution as bar chart
    ax1.bar(
        start_years.index,
        start_years.values,
        color="green",
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_title("Ticker Start Year Distribution")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Tickers")
    ax1.axvline(
        start_years.index[start_years.argmax()],
        color="red",
        linestyle="--",
        label=f"Mode: {start_years.index[start_years.argmax()]}",
    )
    ax1.legend()

    # Plot end date distribution as bar chart
    ax2.bar(
        end_years.index, end_years.values, color="red", alpha=0.7, edgecolor="black"
    )
    ax2.set_title("Ticker End Year Distribution")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Number of Tickers")
    ax2.axvline(
        end_years.index[end_years.argmax()],
        color="darkred",
        linestyle="--",
        label=f"Mode: {end_years.index[end_years.argmax()]}",
    )
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_data_quality(
    daily_tickers: pd.Series, rolling_median_21d: pd.Series = None
) -> None:
    """Visualize data quality: Active tickers over time and distribution"""
    # Create figure with 2 subplots stacked vertically for time series and histogram
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Active tickers over time as line plot
    axes[0].plot(
        daily_tickers.index,
        daily_tickers.values,
        linewidth=0.8,
        alpha=0.7,
        label="Daily Count",
    )

    # Add overall median as reference line (dashed)
    axes[0].axhline(
        daily_tickers.median(),
        color="gray",
        linestyle=":",
        alpha=0.6,
        label=f"Overall median: {daily_tickers.median():.0f}",
    )

    # Add 21-day rolling median so we can see how the numbers are changing.abs
    # and say change a lot !
    if rolling_median_21d is not None:
        axes[0].plot(
            rolling_median_21d.index,
            rolling_median_21d.values,
            color="red",
            linewidth=2,
            linestyle="--",
            label=f"21-day Rolling Median",
        )

    axes[0].set_title("Active Tickers Over Time")
    axes[0].set_ylabel("Number of Tickers")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Distribution as histogram (remains unchanged)
    axes[1].hist(
        daily_tickers.values, bins=50, color="steelblue", edgecolor="black", alpha=0.7
    )
    axes[1].axvline(
        daily_tickers.median(),
        color="red",
        linestyle="--",
        label=f"Median: {daily_tickers.median():.0f}",
    )
    axes[1].set_title("Distribution of daily active tickers")
    axes[1].set_xlabel("Active Tickers")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_gap_analysis(gap_stats: pd.DataFrame, GAP_THRESHOLD: float) -> None:
    """Visualize gap distribution and top 10 problematic tickers"""

    # Create figure with 2 subplots side by side for gap analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Create histogram of gap percentages across all tickers
    gap_stats["gap_pct"].hist(
        bins=50, ax=ax1, color="coral", edgecolor="black", alpha=0.7
    )
    ax1.axvline(
        GAP_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {GAP_THRESHOLD}%",
    )
    ax1.set_title("Distribution of InnerGaps (%)")
    ax1.set_xlabel("Gap Percentage")
    ax1.set_ylabel("Number of Tickers")
    ax1.legend()

    # Create horizontal bar chart of top 10 tickers with highest gaps
    top_gaps = gap_stats.nlargest(10, "gap_pct")
    ax2.barh(range(len(top_gaps)), top_gaps["gap_pct"], color="darkred", alpha=0.7)
    ax2.set_yticks(range(len(top_gaps)))
    ax2.set_yticklabels(top_gaps.index)
    ax2.axvline(GAP_THRESHOLD, color="red", linestyle="--", linewidth=2)
    ax2.set_title("Top 10 Tickers with Highest Gap %")
    ax2.set_xlabel("Gap Percentage")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_liquidity_filters(
    ticker_stats: pd.DataFrame,
    daily_counts: pd.DataFrame,
    daily_counts_price: pd.DataFrame,
    daily_counts_vol: pd.DataFrame,
    daily_counts_dvol: pd.DataFrame,
    MIN_PRICE: float,
    MIN_VOL: int,
    MIN_DVOL: int,
) -> None:
    """Visualize liquidity filter results"""
    # Create masks based on ticker statistics to identify qualifying tickers
    mask_price = ticker_stats["med_price"] >= MIN_PRICE
    mask_vol = ticker_stats["med_vol"] >= MIN_VOL
    mask_dvol = ticker_stats["med_dvol"] >= MIN_DVOL
    mask_all = mask_price & mask_vol & mask_dvol

    # Create 2x3 grid of subplots for comprehensive liquidity analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Safe stackplot function to handle dimension mismatches robustly
    def safe_stackplot(ax, df, ok_col, invalid_col, ok_label, invalid_label, title):
        if df.empty or len(df) == 0:
            ax.text(
                0.5,
                0.5,
                "No Data Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            return

        # Extract arrays with explicit length checking
        ok_counts = df.get(ok_col, pd.Series(0, index=df.index))
        invalid_counts = df.get(invalid_col, pd.Series(0, index=df.index))

        # Ensure both arrays have the same length by aligning indices
        common_index = ok_counts.index.intersection(invalid_counts.index)
        if len(common_index) == 0:
            ax.text(
                0.5,
                0.5,
                "No Overlapping Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            return

        ok_counts = ok_counts.loc[common_index]
        invalid_counts = invalid_counts.loc[common_index]

        # Plot with guaranteed same-length arrays
        ax.stackplot(
            common_index,
            ok_counts.values,
            invalid_counts.values,
            labels=[ok_label, invalid_label],
            colors=["green", "red"],
            alpha=0.6,
        )
        ax.set_title(title)
        ax.set_ylabel("Number of Tickers")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    # Plot valid vs invalid tickers over time, using safe stackplot
    safe_stackplot(
        axes[0, 0],
        daily_counts,
        "Valid",
        "Invalid",
        "Valid",
        "Invalid",
        "Valid vs Invalid Tickers Over Time",
    )

    # Plot price filter results over time using safe stackplot
    safe_stackplot(
        axes[0, 1],
        daily_counts_price,
        "Price_OK",
        "Price_Invalid",
        "Price OK",
        "Price Invalid",
        "Price Filter Over Time",
    )

    # Plot volume filter results over time using safe stackplot
    safe_stackplot(
        axes[0, 2],
        daily_counts_vol,
        "Vol_OK",
        "Vol_Invalid",
        "Volume OK",
        "Volume Invalid",
        "Volume Filter Over Time",
    )

    # Plot dollar volume filter results over time using safe stackplot
    safe_stackplot(
        axes[1, 0],
        daily_counts_dvol,
        "DVol_OK",
        "DVol_Invalid",
        "Dollar Vol OK",
        "Dollar Vol Invalid",
        "Dollar Volume Filter Over Time",
    )

    # Create bar chart showing filter status summary
    ax = axes[1, 1]
    counts = [mask_price.sum(), mask_vol.sum(), mask_dvol.sum(), mask_all.sum()]
    labels = ["Price OK", "Vol OK", "$Vol OK", "ALL OK"]
    colors = ["skyblue", "skyblue", "skyblue", "green"]
    bars = ax.bar(labels, counts, color=colors, edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )
    ax.set_title("Liquidity Filter Results")
    ax.set_ylabel("Number of Tickers")
    ax.grid(axis="y", alpha=0.3)

    # Empty field for better layout alignment
    ax = axes[1, 2]
    ax.axis("off")

    plt.tight_layout()
    plt.show()


def print_summary(df_original: pd.DataFrame, df_final: pd.DataFrame) -> None:
    """Print pipeline summary"""

    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE SUMMARY")
    print("=" * 60)

    # Calculate dimensions for comparison
    orig_tickers = df_original.index.get_level_values("ticker").nunique()
    final_tickers = df_final.index.get_level_values("ticker").nunique()

    print(f"\n{'ORIGINAL DATA':<30} {'FINAL DATA':<30}")
    print(f"{'-' * 60}")
    print(
        f"{'Tickers:':<30} {orig_tickers:<15,} → {final_tickers:<15,} ({final_tickers / orig_tickers * 100:.1f}%)"
    )
    print(
        f"{'Rows:':<30} {len(df_original):<15,} → {len(df_final):<15,} ({len(df_final) / len(df_original) * 100:.1f}%)"
    )

    # Calculate time period for comparison
    orig_start = df_original.index.get_level_values("date").min()
    orig_end = df_original.index.get_level_values("date").max()
    final_start = df_final.index.get_level_values("date").min()
    final_end = df_final.index.get_level_values("date").max()

    print(f"\n{'Date Range:':<30} {orig_start} → {orig_end}")
    print(f"{'After Filter:':<30} {final_start} → {final_end}")

    # Calculate and display missing values in final dataset
    print(f"\n{'MISSING VALUES (Final)':<30}")
    print(f"{'-' * 60}")
    for col in df_final.columns:
        missing = df_final[col].isna().sum()
        pct = missing / len(df_final) * 100
        print(f"{col:<20} {missing:>10,} ({pct:>5.2f}%)")

    # Calculate and display  data quality metrics
    daily_tickers = df_final.reset_index().groupby("date")["ticker"].nunique()
    print(f"\n{'DATA QUALITY':<30}")
    print(f"{'-' * 60}")
    print(f"{'Median Tickers/Day:':<30} {daily_tickers.median():.0f}")
    print(
        f"{'Min/Max Tickers/Day:':<30} {daily_tickers.min():.0f} / {daily_tickers.max():.0f}"
    )

    print("\n" + "=" * 60)
