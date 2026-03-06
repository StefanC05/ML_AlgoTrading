"""
Model Utilities

This file provides  functions  for the Models used in train_models
walk-forward validation, performance evaluation, and data preprocessing
Sry for all the comments but i still learning.

Key Components:
1. Time series cross-validation, by walkforward an  (+purge periods)
2. Performance metrics  (RMSE, MAE, Sharpe ratio)
3.Model training steps
4. Results documentation, Hyperparamter, ML Parameter and performance metrics all that stuff
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch


# %%
# ----------------------------static variable-----------------------------------------------------------------------

#  TIME SERIES VALIDATION CONFIGURATION ===
# Rolling Walk-Forward mit Purgatory für vollständigen Datensatz 2007-2026
# 7 Jahre Training + 126 Tage (6 Monate) Test mit Rolling-Window

TRAIN_WINDOW_YEARS = 7  # 7 Jahre Training (1764 Trading-Tage)
TEST_SIZE_DAYS = 126  # 6 Monate Test (126 Tage)
PURGE_SIZE_DAYS = 15  # Purge zwischen Training und Test

# Phase 1: Hyperparameter Optimization Period (2007-2017)
# 11 Jahre Daten für Hyperparameter-Optimierung mit 7-Jahres Fenstern
PHASE_1_START = "2007-01-01"
PHASE_1_END = "2017-12-31"
PHASE_1_N_SPLITS = 8  # 8 Splits für Phase 1 (begrenzt für Performance)

# Phase 2: Out-of-Sample Testing Period (2011-2026)
# Letzte 7 Jahre von Phase 1 (2011-2017) + gesamter Zeitraum bis 2026
PHASE_2_START = "2011-01-01"
PHASE_2_END = "2026-12-01"
PHASE_2_N_SPLITS = 12  # 12 Splits für Phase 2 (begrenzt für Performance)


RANDOM_FOREST = "RandomForest"
LSTM = "LSTM"
TCN = "TCN"
TFT = "TFT"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PurgeTimeSeriesSplit:
    """
    This class implements proper contiguous walk-forward validation with fixed rolling training windows,
    following Marco Lopez de Prado's methodology. Each fold follows immediately after the previous one
    with no gaps,ensuring complete coverage of the test period.
    """

    def split(self, df: pd.DataFrame, n_splits: int = 5):
        """
        Generate contiguous rolling windows for walk-forward validation.

        Parameters:
        df : MultiIndex Stock Data  (ticker, date)
        n_splits : Maximum number of walk-forward splits (may be reduced based on available data)

        Yields:
        tuple
            (fold_idx, mask_tr, mask_val, train_end_date,
             test_start_date, test_end_date)
        """
        dates = sorted(df.index.get_level_values("date").unique())
        n_dates = len(dates)

        train_window_days = TRAIN_WINDOW_YEARS * 252  # Trading days per year

        # Start with the first possible training window
        # first_train_start_idx = 0
        first_train_end_idx = train_window_days

        if first_train_end_idx >= n_dates:
            raise ValueError("Not enough data for even one training window")

        fold_idx = 0
        current_test_start_idx = first_train_end_idx

        while fold_idx < n_splits and current_test_start_idx + TEST_SIZE_DAYS < n_dates:
            # Training window ends at test start(minus purge! for sure!)
            train_end_idx = current_test_start_idx - PURGE_SIZE_DAYS
            train_start_idx = max(0, train_end_idx - train_window_days)

            # Test period
            test_start_idx = current_test_start_idx
            test_end_idx = min(test_start_idx + TEST_SIZE_DAYS, n_dates)

            # Get dates for masks
            train_start_date = dates[train_start_idx]
            train_end_date = dates[train_end_idx - 1]  # Exclusive
            test_start_date = dates[test_start_idx]
            test_end_date = dates[test_end_idx - 1]

            # Create boolean masks
            mask_tr = (df.index.get_level_values("date") >= train_start_date) & (
                df.index.get_level_values("date") < train_end_date
            )
            mask_val = (df.index.get_level_values("date") >= test_start_date) & (
                df.index.get_level_values("date") <= test_end_date
            )

            # Check minimum data size requirements
            if mask_tr.sum() > 500 and mask_val.sum() > 50:
                yield (
                    fold_idx,
                    mask_tr,
                    mask_val,
                    train_end_date,
                    test_start_date,
                    test_end_date,
                )
                fold_idx += 1

                # Move to next fold: training ends where this test starts (minus purge)
                current_test_start_idx = test_start_idx + TEST_SIZE_DAYS
            else:
                # Not enough data for this fold, stop here
                break


# 4 PERFORMANCE METRICS
def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculate performance metrics for model evaluation. So after training we can see  which Metric
    are a better prediction , or building a databse vor all test.
    Later adding MAR etc more trading like.

    Returns:  dict       (containing RMSE, MAE, and Sharpe ratio)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))

    # Sharpe Ratio -
    pnl = np.sign(y_pred) * y_true
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0

    return {"rmse": rmse, "mae": mae, "sharpe": sharpe}


# 5. DATA PREPARATION UTILITIES
# ____________________________________


def prepare_data_splits(df: pd.DataFrame, mask_tr, mask_val, features, target) -> tuple:
    """
    df : Full dataset
     mask_tr : Training mask
     mask_val : Validation mask

     Returns:  tuple        (X_train, y_train, X_val, y_val)
    """
    X_train = df.loc[mask_tr, features]
    y_train = df.loc[mask_tr, target]
    X_val = df.loc[mask_val, features]
    y_val = df.loc[mask_val, target]

    return X_train, y_train, X_val, y_val


def train_dataPrep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Later it should use a more robust and felxible way, for winsoring.
    so we are not forcing 2.5% on both sides. We "analyse" the dataset and then cuting it down.
    Way better then delete the hole colums/row when there are to large.
    """
    clip_pattern = r"(ret_|log_ret_|parkinson_|var_ratio_|mom_ratio_|NATR|bbu|bbl|tbm_ret|fwd_vola)"
    clip_cols = df.filter(regex=clip_pattern).columns.tolist()
    if not clip_cols:
        return df.copy()

    df_c = df.copy()
    lower = df_c[clip_cols].quantile(0.025)  # 2.5% quantile as lower bound
    upper = df_c[clip_cols].quantile(0.975)  # 97.5% quantile as upper bound

    for col in clip_cols:
        df_c[col] = df_c[col].clip(lower=lower[col], upper=upper[col])
    return df_c


def identify_targets_and_features(df: pd.DataFrame) -> tuple:
    """
    Identify target and feature columns from dataset. So it can be savely seperatet for ML analysis.
    >> I need a file where i write all names of Targets and features and not hardcoding it.

    df : Full dataset

    Returns:    tuple        (targets, features) lists
    """
    targets = []

    # Specific fwd_log_ret horizons
    for horizon in [1, 3, 5, 10]:
        col = f"fwd_log_ret_{horizon}"  # with zero?
        if col in df.columns:
            targets.append(col)

    # All label and tbm_ret columns
    for col in df.columns:
        if col.startswith(("dl_label_", "tbm_label_", "tbm_ret_")):
            targets.append(col)

    features = [
        c
        for c in df.columns
        if c not in targets and "label" not in c and "ret" not in c
    ]

    return targets, features


def save_results(
    oos_preds_df: pd.DataFrame,
    fold_results: list[dict],
    stats_history: list[dict],
    timing_stats: list[dict],
    models_final: dict,
) -> None:
    """
    For a proper documentation for sure.
    Save all training results in files. Later it will be helpfull to understand why we choos which paramter.

        oos_preds_df :      Out-of-sample predictions
        fold_results :          Detailed fold results
        stats_history :      Aggregated statistics
        timing_stats :      Timing information
        models_final :      Final trained models
    """
    import joblib

    # 1. Predictions
    pred_cols = [c for c in oos_preds_df.columns if c.startswith("pred_")]
    oos_preds_final = oos_preds_df[oos_preds_df[pred_cols].notna().any(axis=1)]
    oos_preds_final.to_hdf("results/oos_preds.h5", key="predictions", mode="w")
    print(f"Predictions: results/oos_preds.h5 ({len(oos_preds_final):,} rows)")

    # 2. Per-Fold Detailed Results
    df_fold_results = pd.DataFrame(fold_results)
    df_fold_results.to_csv("results/training_fold_results.csv", index=False)
    print(
        f" Per-fold results. FIle: results/training_fold_results.csv ({len(df_fold_results)} )"
    )

    # 3. Summary Statistics
    df_stats = pd.DataFrame(stats_history)
    summary_stats = (
        df_stats.groupby(["Model", "Target"])
        .agg(
            {
                "RMSE": ["mean", "std", "min", "max"],
                "MAE": ["mean", "std", "min", "max"],
                "Sharpe": ["mean", "std", "min", "max"],
            }
        )
        .round(4)
    )
    summary_stats.columns = [
        "_".join(col).strip() for col in summary_stats.columns.values
    ]
    summary_stats = summary_stats.reset_index()

    summary_stats.to_csv("results/training_stats_summary.csv", index=False)
    print(f" \n Summary statistics save: results/training_stats_summary.csv")

    df_stats.to_csv("results/training_stats_per_fold.csv", index=False)
    print(f"\n Per-fold statistics saved: results\training_stats_per_fold.csv")

    df_timing = pd.DataFrame(timing_stats)
    df_timing.to_csv("results/training_timing_stats.csv", index=False)
    print(f"\n Timing statistics saved: results/training_timing_stats.csv")

    joblib.dump(models_final, "models/models_final.joblib")
    print(
        f" Models saved: models/models_final.joblib ({len(models_final)} models from last fold)"
    )


def train_and_evaluate_model(
    model_type: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series,
    target: str,
    fold_idx: int,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    features: list[str],
    oos_preds_df: pd.DataFrame,
    mask_val: pd.Series | np.ndarray,
    timing_stats: list[dict],
    models_final: dict,
    fold_results: list[dict],
    stats_history: list[dict],
    n_splits: int,
) -> None:
    """
    Unified function to train and evaluate either Random Forest or a NN model (TCN is half working....).

    Parameters
    ----------
    model_type : Either RANDOM_FOREST or LSTM
    X_train, y_train : Training data
    X_val, y_val : Validation data
    target + features : List of feature /Target col name
    fold_idx : Current fold index
    train_end, test_start, test_end : Date information for logging

    oos_preds_df : DataFrame to store out-of-sample predictions
    mask_val : Boolean mask for validation data
    timing_stats : List to store timing information

    models_final  fold_results : self explained.TO write it in a file.
    stats_history : List to store aggregated statistics

    ! No return! Non..Nothing....
    """
    tgt_start = time.time()
    original_size = len(X_train)

    # Handle both pandas DataFrame (for RandomForest) and numpy array (LSTM/TCN ?) for NaN checking
    if isinstance(X_train, np.ndarray):
        valid = y_train.notna() & (~np.isnan(X_train).any(axis=1))
    else:
        valid = y_train.notna() & X_train.notna().all(axis=1)
    X_tr_cl = X_train[valid]
    y_tr_cl = y_train[valid]
    dropped = original_size - len(X_tr_cl)

    # Only print for first fold to avoid spam
    if dropped > 0 and fold_idx == 0:
        print(
            f"Dropped {dropped} rows with NaN values ({dropped / original_size * 100:.1f}%)"
        )

    if len(X_tr_cl) < 500:
        print(f"  Skip {target}: insufficient data")
        timing_stats.append(
            {
                "fold": fold_idx,
                "target": target,
                "model": model_type,
                "time_s": time.time() - tgt_start,
            }
        )
        return

    # Import here to avoid circular imports
    import src.random_forest_model
    import src.lstm_model
    import src.tcn_model
    import src.tft_model

    # Choose appropriate training function based on model type
    if model_type == RANDOM_FOREST:
        train_predict_func = src.random_forest_model.train_and_predict_rf
        pred_col_prefix = "pred_RF_"
        model_key_prefix = "RF_"
    elif model_type == LSTM:
        train_predict_func = src.lstm_model.train_and_predict_lstm
        pred_col_prefix = "pred_LSTM_"
        model_key_prefix = "LSTM_"
    elif model_type == TCN:
        train_predict_func = src.tcn_model.train_and_predict_tcn
        pred_col_prefix = "pred_TCN_"
        model_key_prefix = "TCN_"
    elif model_type == TFT:
        train_predict_func = src.tft_model.train_and_predict_tft
        pred_col_prefix = "pred_TFT_"
        model_key_prefix = "TFT_"

    # Train and predict
    predictions, best_params, trained_model = train_predict_func(
        X_tr_cl, y_tr_cl, X_val, fold_idx
    )

    # Check for NaN values in predictions
    if np.isnan(predictions).any():
        print(
            f" Warning:NaN values detected in {model_type} predictions for {target}. Skipping metrics calculation."
        )
        # Skip this target if predictions contain NaN
        timing_stats.append(
            {
                "fold": fold_idx,
                "target": target,
                "model": model_type,
                "time_s": time.time() - tgt_start,
            }
        )
        return

    # Save predictions
    col_name = f"{pred_col_prefix}{target}"
    oos_preds_df.loc[mask_val, col_name] = predictions

    # Save last fold model
    if fold_idx == n_splits - 1:
        models_final[f"{model_key_prefix}{target}"] = trained_model

    # metrics
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = np.mean(np.abs(y_val - predictions))
    pnl = np.sign(predictions) * y_val
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0

    # statistics per fold
    fold_result = {
        "Model": model_type,
        "Fold": fold_idx + 1,
        "Train_End": str(train_end.date()),
        "Test_Start": str(test_start.date()),
        "Test_End": str(test_end.date()),
        "Target": target,
        "Train_Size": len(X_tr_cl),
        "Test_Size": len(X_val),
        "Features_N": len(features),
        "Hyperparameters": str(best_params),
        "RMSE": round(rmse, 5),
        "MAE": round(mae, 5),
        "Sharpe": round(sharpe, 4),
    }
    fold_results.append(fold_result)

    # Aggregate stats
    stats_history.append(
        {
            "Model": model_type,
            "Fold": fold_idx + 1,
            "Target": target,
            "RMSE": round(rmse, 5),
            "MAE": round(mae, 5),
            "Sharpe": round(sharpe, 4),
        }
    )

    # Progress print
    print(f"    {model_type[:4]} | {target[:15]:15s} | Sharpe: {sharpe:+.2f}")

    timing_stats.append(
        {
            "fold": fold_idx,
            "target": target,
            "model": model_type,
            "time_s": time.time() - tgt_start,
        }
    )


def print_summary_report(
    stats_history: list[dict], targets: list, n_splits: int
) -> None:
    """
    Print a summary report of training results. all of them...every time.
        stats_history : Training statistics history
        targets :           List of target variables
        n_splits : Number of walk-forward splits
    """
    if not stats_history:
        print("No training statistics available.")
        return

    df_stats = pd.DataFrame(stats_history)

    # Calculate summary stats per modeland for all targets
    summary_stats = (
        df_stats.groupby(["Model", "Target"])
        .agg(
            {
                "RMSE": ["mean", "std", "min", "max"],
                "MAE": ["mean", "std", "min", "max"],
                "Sharpe": ["mean", "std", "min", "max"],
            }
        )
        .round(4)
    )
    summary_stats.columns = [
        "_".join(col).strip() for col in summary_stats.columns.values
    ]
    summary_stats = summary_stats.reset_index()

    print("TRAINING SUMMARY \n")
    print("----" * 25)
    print(f"\nSplits: {n_splits}")
    print(f"Test Period: {TEST_SIZE_DAYS} Tage, Purge: {PURGE_SIZE_DAYS} Tage")

    print("\n---- Performance Summary (Mean ± Std) ---")
    for _, row in summary_stats.iterrows():
        print(f" \n {row['Model']:12s} | {row['Target']:20s}")
        print(
            f"  Sharpe: {row['Sharpe_mean']:+.3f} ± {row['Sharpe_std']:.3f} (range: {row['Sharpe_min']:+.3f} to {row['Sharpe_max']:+.3f})"
        )
        print(f" RMSE: {row['RMSE_mean']:.5f} ± {row['RMSE_std']:.5f}")
        # print(f" MAE:  {row['MAE_mean']:.5f} ± {row['MAE_std']:.5f}")

    print("TRAINING COMPLETE")
    # print(f" MAE:  {row['MAE_mean']:.5f} ± {row['MAE_std']:.5f}")
