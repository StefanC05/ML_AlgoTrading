"""
TFT Model - DARTS Implementation (MINIMAL VERSION)
==================================================
Drastically simplified for stable training.

12.02
I have still problems mit NaN and non trimmed Data.
In a later version , this should be handeld in preprocess.py !
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from darts.models import TFTModel
from darts import TimeSeries
from sklearn.preprocessing import StandardScaler

from src.model_utils import DEVICE


#
def train_and_predict_tft(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    fold_idx: int,
    trials: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any], TFTModel]:
    """
    Train TFT - MINIMAL VERSION for debugging.
    """
    print(f"Training TFT for fold {fold_idx + 1}...")

    # MINIMAL configuration -  more advanced condifuration should be testet after errorhandling was moved to preprocess.py
    best_params: Dict[str, Any] = {
        "input_chunk_length": 5,  # Reduced from  15
        "output_chunk_length": 1,
        "hidden_size": 32,  # DRASTICALLY reduced from 384
        "hidden_continuous_size": 16,  # Reduced from 192
        "num_attention_heads": 2,  # Reduced from 12
        "lstm_layers": 1,  # Reduced from 2
        "dropout": 0.1,
        "batch_size": 32,  # Reducedfrom 96
        "n_epochs": 5,  # Test with just 5 epochs
        "optimizer_kwargs": {
            "lr": 0.001,
        },
        "random_state": 42,
        "add_relative_index": True,
        "use_static_covariates": False,
        "pl_trainer_kwargs": {
            "accelerator": "cpu",  # Force CPU for stability, CUDA not possible i just have a AMD Radeon
            "devices": 1,
            "enable_progress_bar": True,
        },
    }

    # Validate data
    print(f"  Data stats - y_train: {len(y_train)} samples")
    print(f"  y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"  y_train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")

    # Handle NaN/Inf
    y_train_clean = y_train.replace([np.inf, -np.inf], np.nan).fillna(0)

    # CRITICAL FIX: Clip extreme outliers before scaling  # 12.02 for later version clipping should be more Data oriented
    lower = y_train_clean.quantile(0.05)  # 0.15% percentile
    upper = y_train_clean.quantile(0.95)  # 95% percentile
    y_train_clipped = y_train_clean.clip(lower, upper)
    print(f"  Clipped to [{lower:.4f}, {upper:.4f}]")

    # Check zero variance
    if y_train_clipped.std() < 1e-10:
        print(f"  WARNING: Target has near-zero variance! Using zeros.")
        return np.zeros(len(X_val)), best_params, None

    # Scale targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(
        y_train_clipped.values.reshape(-1, 1)
    ).flatten()

    print(f"  Scaled range: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")

    # Create TimeSeries
    train_series = TimeSeries.from_values(y_train_scaled)

    try:
        # Initialize model
        model = TFTModel(**best_params)
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
        print(f"ERROR during training: {e}")
        print(f" Falling back to ZERO predictions! ")
        predictions = np.zeros(len(X_val))
        model = None

    # Handle NaN
    if np.isnan(predictions).any():
        print("Warning: NaN in predicktions, replacing with zeros")
        predictions = np.nan_to_num(predictions, nan=0.0)

    predictions = predictions[: len(X_val)]

    return predictions, best_params, model
