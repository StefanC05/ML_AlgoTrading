"""
TCN Model - DARTS Implementation (MINIMAL VERSION)
    Drastically simplified for stable training.

13.02  TCN need the most time.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from darts.models import TCNModel
from darts import TimeSeries
from sklearn.preprocessing import StandardScaler

from src.model_utils import DEVICE


def train_and_predict_tcn(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    fold_idx: int,
    trials: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any], TCNModel]:
    """
    Train TCN - MINIMAL VERSION for debugging.
    """
    print(f"Training TCN for fold {fold_idx + 1}...")

    # MINIMAL configuration - will be increased once stable
    best_params: Dict[str, Any] = {
        "input_chunk_length": 5,  # Reduced from 10
        "output_chunk_length": 1,
        "num_filters": 16,  # DRASTICALLY reduced from 128
        "num_layers": 1,  # Reduced from 3
        "kernel_size": 2,  # Reduced from 3
        "dilation_base": 2,
        "dropout": 0.1,  # Reduced
        "batch_size": 32,  # Reduced from 128
        "n_epochs": 5,  # Test with just 5 epochs first
        "optimizer_kwargs": {
            "lr": 0.001,  # Standard learningrate
        },
        "random_state": 42,
        "pl_trainer_kwargs": {
            "accelerator": "cpu",  # Force CPU for stability
            "devices": 1,
            "enable_progress_bar": True,
        },
    }

    # Validate data before processing
    print(f"  Data stats - y_train: {len(y_train)} samples")
    print(f"  y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"  y_train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")

    # Handle NaN/Inf in target data
    y_train_clean = y_train.replace([np.inf, -np.inf], np.nan).fillna(0)

    # CRITICAL FIX: Clip extreme outliers before scaling, schould be moved to preprocess.py ( 18.02)
    # fwd_log_ret can have extreme values (-2.3, +2.3) that become -48/+48 sigma
    lower = y_train_clean.quantile(0.05)  # 5% percentile
    upper = y_train_clean.quantile(0.95)  # 95% percentile
    y_train_clipped = y_train_clean.clip(lower, upper)
    print(f"  Clipped to [{lower:.4f}, {upper:.4f}]")

    # Check if data has zero variance
    if y_train_clipped.std() < 1e-10:
        print(f"WARNING: Target has near-zero variance!Using zeros.")
        return np.zeros(len(X_val)), best_params, None

    # Scale targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(
        y_train_clipped.values.reshape(-1, 1)
    ).flatten()

    print(f"  Scaled range: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")

    train_series = TimeSeries.from_values(y_train_scaled)

    try:
        model = TCNModel(**best_params)
        model.fit(train_series, verbose=True)

        n_val = len(X_val)
        predictions_scaled = (
            model.predict(n=n_val, series=train_series).values().flatten()
        )

        # Inverse transform
        predictions = target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()

        print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    except Exception as e:
        print(f" ERROR during training: {e}")
        print(f" Falling back to zero predictions")
        predictions = np.zeros(len(X_val))
        model = None

    # Handle NaN  08.02 still Data problems....working handeling should be moves to preprocess.py
    if np.isnan(predictions).any():
        print("Warning: NaN in predictions, replacing with zeros")
        predictions = np.nan_to_num(predictions, nan=0.0)

    predictions = predictions[: len(X_val)]

    return predictions, best_params, model
