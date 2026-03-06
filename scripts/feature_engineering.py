# 04_FeatureEngineering_EDA.py
# =============================================================================
# AUFGABE: Feature Creation & EDA

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import sys
from pathlib import Path

# Add project root to Python path to enable src imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from src.feature_lib import (
    add_basic_features,
    add_complex_features,
    add_targets,
    cleanup_features,
    create_global_features,
)
from src.model_utils import identify_targets_and_features

IMPORT_FILE = "data/interim/DATA_02_assets_clean.h5"
OUTPUT_FILE = "data/processed/DATA_03_features_targets.h5"
INTERVALS = [1, 3, 5, 10]  # ten just for "longer" features.
IN_SAMPLE_END = "2016-12-31"  # Angepasst an Phase 1 Ende
GMM_COMPONENTS = 3  # Konfigurierbar
PCA_PIVOT_ROW = "ret_5"


def perform_mi_analysis(
    df: pd.DataFrame,
    target_horizon: str = "fwd_log_ret_5",
    sample_size: int = 10000,  # not needed for daily data but for intraday
) -> pd.DataFrame:
    """
    Perform Mutual Information analysis so we will find non-linear feature correlation.
    In the original, Stefan Jansen does a Spearman correlation, but this method will find only linear correlation.

            df :         df for:  features & targets
            target_horizon : default "fwd_log_ret_5"        Target variable to analyze relationships with
            sample_size :def 10000              Number of samples to use for MI calculation (for speed)

    Returns: mi_df
    """
    print(
        "Starting Mutual Information (MI) analysis"
    )  # in a future far far away: denoising(no pca) so we group the feature.

    # Select features (exclude targets and labels)
    features = [
        c
        for c in df.columns
        if "fwd_" not in c and "tbm_" not in c and "label" not in c and "dl_" not in c
    ]

    # Check if target exists    extern
    if target_horizon not in df.columns:
        print(f" Warning: '{target_horizon}' not found. Skipping MI analysis.")
        return pd.DataFrame()

    print(f"MI for target: {target_horizon}")
    print(f"Number of Features to analyze: {len(features)}")
    print(f"sample size: {sample_size}")

    df_sample = df.sample(
        min(sample_size, len(df)), random_state=42
    )  # full data , is fast enough

    # Calculate MI scores for each feature -> target
    # should be
    mi_scores = []
    target_data = df_sample[target_horizon].dropna()
    valid_idx = target_data.index

    for feature in features:
        try:
            feature_data = df_sample.loc[valid_idx, feature].dropna()
            common_idx = feature_data.index.intersection(target_data.index)

            if len(common_idx) < 100:  # nötig?
                print(f" Warning: Not enough data for {feature}")
                continue

            X = feature_data.loc[common_idx].values.reshape(-1, 1)
            y = target_data.loc[common_idx].values

            # Clean data: remove inf and extreme values
            valid_mask = np.isfinite(X.flatten()) & np.isfinite(y)
            if np.sum(valid_mask) < 100:
                print(f" Warning: Not enough finite data for {feature}")
                continue

            X_clean = X[valid_mask].reshape(-1, 1)
            y_clean = y[valid_mask]

            X_clean = np.clip(X_clean, -1e10, 1e10)
            y_clean = np.clip(y_clean, -1e10, 1e10)

            mi = mutual_info_regression(X_clean, y_clean, random_state=42)[0]
            mi_scores.append(
                {
                    "feature": feature,
                    "target": target_horizon,
                    "mi_score": mi,
                }
            )
        except Exception as e:
            print(f" Warning: Could not calculate MI for {feature}: {e}!")
            continue

    mi_df = pd.DataFrame(mi_scores).sort_values("mi_score", ascending=False)
    # Save results
    mi_df.to_csv("results/mi_feature_scores.csv", index=False)
    mi_df.to_hdf("results/mi_feature_scores.h5", key="mi_scores", mode="w")

    print(
        "MI scores saved in: results/mi_feature_scores.csv and results/mi_feature_scores.h5"
    )

    # Create visualization....still data science here....
    # # Top 20 features by MI score
    plt.figure(figsize=(12, 8))

    top_features = mi_df.head(20)
    plt.subplot(2, 1, 1)
    bars = plt.barh(range(len(top_features)), top_features["mi_score"])
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Mutual Information Score")
    plt.title(f"Top 20  (Target: {target_horizon})")
    plt.gca().invert_yaxis()

    # Color bars based on score strength
    max_mi = mi_df["mi_score"].max()
    for i, bar in enumerate(bars):
        score = top_features.iloc[i]["mi_score"]
        if score > max_mi * 0.8:
            bar.set_color("darkgreen")
        elif score > max_mi * 0.5:
            bar.set_color("orange")
        elif score > max_mi * 0.2:
            bar.set_color("lightcoral")
        else:
            bar.set_color("lightgray")

    # MI score distribution
    plt.subplot(2, 1, 2)
    plt.hist(mi_df["mi_score"], bins=30, alpha=0.7, edgecolor="black")
    plt.xlabel("MI Scores")
    plt.ylabel("Frequency")
    plt.title("Distribution of MI Scores")

    plt.tight_layout()
    plt.savefig("reports/figures/mi_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary
    print("\n MI Summary:")
    print(f" Total features analyzed: {len(mi_df)}")
    if len(mi_df) > 0:
        # print(f" Max MI score: {mi_df['mi_score'].max():.4f}") only sense per feature
        #  print(f" Mean MI score: {mi_df['mi_score'].mean():.4f}")
        print("top 5 features by MIScore:")
        for i, row in mi_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['mi_score']:.4f}")

    return mi_df


def compute_mi_matrix(
    df: pd.DataFrame,
    features: list,
    sample_size: int = 5000,  # not needed for daily data but for intraday
) -> pd.DataFrame:
    """df :        feature df
        features :         List of feature columns
        sample_size :  for calc should become überflüssig
    Returns: df     MI matrix
    """
    print(
        f" Computing MI matrix for {len(features)} features using sample of {sample_size}"
    )

    df_sample = df[features].dropna().sample(min(sample_size, len(df)), random_state=42)
    mi_matrix = pd.DataFrame(index=features, columns=features)

    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i <= j:  # Symmetric, compute once
                try:
                    x = df_sample[feat1].values
                    y = df_sample[feat2].values

                    # Clean data: remove inf and extreme values
                    valid_mask = np.isfinite(x) & np.isfinite(y)
                    if np.sum(valid_mask) < 100:
                        mi_matrix.loc[feat1, feat2] = 0
                        if i != j:
                            mi_matrix.loc[feat2, feat1] = 0
                        continue

                    x_clean = np.clip(x[valid_mask], -1e10, 1e10)
                    y_clean = np.clip(y[valid_mask], -1e10, 1e10)

                    mi = mutual_info_regression(x_clean.reshape(-1, 1), y_clean)[0]
                    mi_matrix.loc[feat1, feat2] = mi
                    if i != j:
                        mi_matrix.loc[feat2, feat1] = mi
                except Exception as e:
                    print(f" Error computing MI for {feat1}-{feat2}: {e}")
                    mi_matrix.loc[feat1, feat2] = 0
                    if i != j:
                        mi_matrix.loc[feat2, feat1] = 0

    return mi_matrix.astype(float)


# %%
#   Main()
def main() -> None:
    print("--- 1. LOAD DATA ---")
    try:
        prices = pd.read_hdf(IMPORT_FILE, key="assets_clean")
    except:
        print(f"Error: {IMPORT_FILE} not found! ")
        return

    print("\n-----2. RUN FEATURE ENGINEERING (pre engineering for all)----")
    features = create_global_features(prices, INTERVALS, GMM_COMPONENTS)

    # Drop columns with >50% NaN (poor quality features), bye miss calculations or wrong type of calculation...needed? early testing??
    thresh_col = int(0.5 * len(features))
    initial_cols = features.shape[1]
    print(
        f"Date range before column drop: {features.index.get_level_values('date').min()} to {features.index.get_level_values('date').max()}"
    )
    features = features.dropna(axis=1, thresh=thresh_col)
    dropped_cols = initial_cols - features.shape[1]
    print(
        f"Dropped {dropped_cols} columns with >50% NaN, remaining columns: {features.shape[1]}"
    )
    print(
        f"Date range after column drop: {features.index.get_level_values('date').min()} to {features.index.get_level_values('date').max()}"
    )

    # features = train_dataPrep(features)  # Winsorizing (nur Train, aber hier global - passe bei Split an)

    # --- NaN HANDLING: Drop rows with NaN in features, keep rows with NaN only in targets ---
    # This removes rows where lags/rolling calcs  NaN
    # Targets can have NaN at end due to forward-looking horizons

    print(f"\n ------NaN HANDLING,  of rolling NaN's")
    print(f" shape>>Before NaN removal: {features.shape}")
    print(f" Total NaN values: {features.isna().sum().sum()}")

    targets, features_cols = identify_targets_and_features(features)
    print(f"Identified {len(features_cols)} features and {len(targets)} targets")

    rows_with_nan_features = features[features_cols].isna().any(axis=1).sum()
    print(f"Rows with NaN in features: {rows_with_nan_features}")

    # Drop rows where features have NaN (but keep if only targets have NaN)
    features_clean = features.dropna(subset=features_cols)
    print(
        f" \n DF shape AFTER NaN removal: {features_clean.shape} -------------------------------------"
    )
    print(f"Rows removed: {features.shape[0] - features_clean.shape[0]}")

    # Calculate percentage of data removed
    removal_pct = (
        (features.shape[0] - features_clean.shape[0]) / features.shape[0] * 100
    )
    print(f"Data removed percentage: {removal_pct:.2f}%")
    print(
        f"Date range after row drop: {features_clean.index.get_level_values('date').min()} to {features_clean.index.get_level_values('date').max()}"
    )

    print(f"Created. Shape: {features_clean.shape}")
    features_clean.to_hdf(OUTPUT_FILE, key="data", mode="w")
    print("Saved to features_targets.h5")

    print(f"\n--------------3. EDA (In-Sample < {IN_SAMPLE_END})-----------------")
    df_eda = features.loc[features.index.get_level_values("date") < IN_SAMPLE_END]
    print(
        f"EDA date range: {df_eda.index.get_level_values('date').min()} to {df_eda.index.get_level_values('date').max()}"
    )

    if df_eda.empty:
        print(" No in-sample data.")
        return

    # For EDA, drop rows with NaN in target or features to ensure valid analysis
    target = "fwd_log_ret_05"

    # Drop NaN only in target (MI can handle NaN in features)
    valid_eda = df_eda.dropna(subset=[target])
    if valid_eda.empty:
        print(
            f" No valid data for MI analysis after dropping NaN in target '{target}'."
        )
    else:
        print(
            f" Using {len(valid_eda):,} rows for MI analysis after dropping NaN in target."
        )
        print(
            f"MI data date range: {valid_eda.index.get_level_values('date').min()} to {valid_eda.index.get_level_values('date').max()}"
        )
        perform_mi_analysis(valid_eda, target)

    # B. Mutual Information Matrix
    print("Mutual Information Matrix")
    feats = [
        c
        for c in df_eda.columns
        if "fwd_" not in c and "tbm_" not in c and "label" not in c
    ]

    # MI matrix for subset of features
    subset_size = min(20, len(feats))  # Limit to 20 features for speed
    selected_feats = feats[:subset_size]
    mi_matrix = compute_mi_matrix(df_eda, selected_feats, sample_size=5000)

    plt.figure(figsize=(12, 10))
    sns.heatmap(mi_matrix, cmap="viridis", square=True, annot=False)
    plt.title(f"Feature Mutual Information Matrix (Subset of {subset_size} features)")
    plt.show()

    # D. New Features Check
    if "tbm_label_05" in df_eda.columns:
        print("\n Triple Barrier Labels found.")
    if "dl_label_05" in df_eda.columns:
        print(" DL Labels found.")


if __name__ == "__main__":
    main()
