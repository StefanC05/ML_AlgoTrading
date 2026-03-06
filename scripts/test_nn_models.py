"""
Test NN Models Individually

Quick test script to validate each NN model (LSTM, TCN, TFT) works with real project data.
Uses only 1 training fold and 1 test fold for fast validation.

Usage:
    python scripts/test_nn_models.py --model lstm
    python scripts/test_nn_models.py --model tcn
    python scripts/test_nn_models.py --model tft
"""

import pandas as pd
import numpy as np
import argparse
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Project imports
project_root = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(project_root))

from src.model_utils import (
    identify_targets_and_features,
    train_dataPrep,
    PurgeTimeSeriesSplit,
)
from src.feature_lib import create_per_fold_features
from src.lstm_model import train_and_predict_lstm
from src.tcn_model import train_and_predict_tcn
from src.tft_model import train_and_predict_tft

IMPORT_FILE = "data/processed/DATA_03_features_targets.h5"


def test_single_model(model_type: str, max_targets: int = 3):
    """
    Test a single NN model with real project data.

    Args:
        model_type: 'lstm', 'tcn', or 'tft'
        max_targets: Maximum number of targets to test (for speed)
    """
    print(f"Testing {model_type.upper()} Model")
    print("=" * 50)

    # Load data
    print("Loading data...")
    try:
        df = pd.read_hdf(IMPORT_FILE, key="data").sort_index()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Identify targets and features
    targets, features = identify_targets_and_features(df)
    targets = targets[:max_targets]  # Limit for testing speed
    print(
        f"Testing with {len(features)} features and {len(targets)} targets: {targets}"
    )

    # Get first training fold and test fold for quick testing
    splitter = PurgeTimeSeriesSplit()
    folds = list(splitter.split(df, n_splits=2))

    if len(folds) < 2:
        print("Not enough data for 2 folds")
        return

    fold_idx, mask_tr, mask_val, train_end, test_start, test_end = folds[0]

    print(f"Using fold {fold_idx + 1}")
    print(f"Training period: {train_end.date()}")
    print(f"Test period: {test_start.date()} to {test_end.date()}")

    # Data preprocessing
    print("\nPreprocessing data...")
    X_tr_raw = df.loc[mask_tr, features]
    X_tr_winsor = train_dataPrep(X_tr_raw)
    X_val_raw = df.loc[mask_val, features]

    # GMM clustering features
    X_tr_winsor, X_val_enhanced = create_per_fold_features(
        X_tr_winsor, X_val_raw, gmm_components=3
    )
    X_val_raw = X_val_enhanced

    # Scale data for NN models
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_winsor)
    X_val_scaled = scaler.transform(X_val_raw)

    print(f"Training data shape: {X_tr_scaled.shape}")
    print(f"Validation data shape: {X_val_scaled.shape}")

    # Test each target
    results = []
    total_start_time = time.time()

    for i, target in enumerate(targets):
        print(f"\nTesting target {i + 1}/{len(targets)}: {target}")

        y_train = df.loc[mask_tr, target]
        y_val = df.loc[mask_val, target]

        target_start = time.time()

        try:
            # Choose model function
            if model_type == "lstm":
                predictions, params, model = train_and_predict_lstm(
                    X_tr_scaled, y_train, X_val_scaled, fold_idx=0
                )
            elif model_type == "tcn":
                predictions, params, model = train_and_predict_tcn(
                    X_tr_scaled, y_train, X_val_scaled, fold_idx=0
                )
            elif model_type == "tft":
                predictions, params, model = train_and_predict_tft(
                    X_tr_scaled, y_train, X_val_scaled, fold_idx=0
                )
            else:
                print(f"Unknown model type: {model_type}")
                continue

            target_time = time.time() - target_start

            # Calculate metrics
            from sklearn.metrics import mean_squared_error

            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            mae = np.mean(np.abs(y_val - predictions))
            pnl = np.sign(predictions) * y_val
            sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0

            print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, Sharpe: {sharpe:.4f}")
            print(f"   Time: {target_time:.2f}s")

            results.append(
                {
                    "target": target,
                    "rmse": rmse,
                    "mae": mae,
                    "sharpe": sharpe,
                    "time_seconds": target_time,
                    "predictions_shape": predictions.shape,
                }
            )

        except Exception as e:
            print(f"Error testing {target}: {e}")
            continue

    total_time = time.time() - total_start_time

    # Summary
    print(f"\n{model_type.upper()} Test Summary")
    print("=" * 30)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Targets tested: {len(results)}")

    if results:
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_rmse = np.mean([r["rmse"] for r in results])
        avg_time = np.mean([r["time_seconds"] for r in results])

        print(f"Average Sharpe: {avg_sharpe:.4f}")
        print(f"Average RMSE: {avg_rmse:.6f}")
        print(f"Average Time: {avg_time:.3f}s")

        # Detailed results
        print("\nDetailed Results:")
        for r in results:
            print(
                f"  {r['target']}: Sharpe={r['sharpe']:.4f}, RMSE={r['rmse']:.4f}, Time={r['time_seconds']:.2f}s"
            )

        print("\nTest completed successfully!")
    else:
        print("No successful tests")


def main():
    parser = argparse.ArgumentParser(description="Test NN models individually")
    parser.add_argument(
        "--model", required=True, choices=["lstm", "tcn", "tft"], help="Model to test"
    )
    parser.add_argument(
        "--targets", type=int, default=3, help="Maximum number of targets to test"
    )

    args = parser.parse_args()

    print("NN Model Testing Script")
    print("=" * 50)

    test_single_model(args.model, args.targets)


if __name__ == "__main__":
    main()
