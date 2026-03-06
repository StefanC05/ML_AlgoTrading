"""
-  Hyperparameter optimization
- Model training
- Prediction generation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


def optimize_rf(X: pd.DataFrame, y: pd.Series) -> dict:
    """
     hyperparameters by GridSearchCV. #later maybe optuna....only for feature and the feature parameter?
        X : Training features
        y : Target values

    Returns    dict        Best hyperparameters found
    """
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 6, 9, 12],
        "min_samples_leaf": [5, 20, 35, 50],  # five to less?
        "max_features": [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    # Speed-Up: Subsample if dataset is large i think it could handel, all data. but if i want use intraday data . i think is an option.....
    if len(X) > 3000:
        idx = np.arange(len(X) - 3000, len(X))
        X_s, y_s = X.iloc[idx], y.iloc[idx]
    else:
        X_s, y_s = X, y

    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        RandomForestRegressor(n_jobs=4, random_state=42),
        param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=4,
    )

    grid_search.fit(X_s, y_s)

    return grid_search.best_params_


def train_rf_model(
    X_train: pd.DataFrame, y_train: pd.Series, best_params: dict
) -> object:
    """
        X_train : Training features
        y_train : Training targets
        best_params
    Returns:    sklearn.pipeline.Pipeline
    """
    rf_pipe = Pipeline(
        [("rf", RandomForestRegressor(**best_params, n_jobs=4, random_state=42))]
    )
    rf_pipe.fit(X_train, y_train)

    return rf_pipe


def train_and_predict_rf(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, fold_idx: int
) -> tuple:
    """
    X_Train: Training features
    X_val: Validation features
    y_train: Training targets
    fold_idx: Current fold index

    Returns:       (predictions, best_params, trained_model)
    """
    best_params = optimize_rf(X_train, y_train)
    model = train_rf_model(X_train, y_train, best_params)
    predictions = model.predict(X_val)

    return predictions, best_params, model
