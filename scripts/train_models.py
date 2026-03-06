"""
Model Training Pipeline

This File , finally, applying ML algos to the Data *cheering*, from loading data to evaluation metrics.
for both Random Forest and TCN (in work, sry) models using purgatory walk-forward.


- Walk-forward cross-validation with purge periods
- hyperparameter optimization
- Performance evaluation
- Results saving and summarization

####################################################
Projects you will carry out:
Extracting and graphically representing financial data.
Preparing data, creating charts, and building
Training and comparing machine learning models

python scripts/train_models.py --phase2 --model 3

So you can split up the time consuming phase1 and phase2 with out loosing configuration Parameter
specily TCN need a long time, even with very simple configuration !
"""

# %%         ############################################################################
import pandas as pd
import time
import sys
import argparse
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import src.model_utils
from src.model_utils import (
    RANDOM_FOREST,
    identify_targets_and_features,
    train_dataPrep,
    PurgeTimeSeriesSplit,
    train_and_evaluate_model,
    save_results,
    print_summary_report,
)
from src.feature_lib import create_per_fold_features

import src.random_forest_model

import src.lstm_model
import src.tcn_model
import src.tft_model

IMPORT_FILE = "data/processed/DATA_03_features_targets.h5"


def train_models_fold(
    phase_num: int,
    fold_idx: int,
    df: pd.DataFrame,
    mask_tr: pd.Series,
    mask_val: pd.Series,
    features: list[str],
    targets: list[str],
    oos_preds_df: pd.DataFrame,
    models_final: dict,
    fold_results: list[dict],
    stats_history: list[dict],
    timing_stats: list[dict],
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    n_splits: int,
    model_selection: int,
) -> None:
    """
    Train all 4 models (RF, LSTM, TCN, TFT) for a single fold.

    fold_idx : Current fold index
    df : Full dataset for this fold
    mask_tr : Training data mask
    mask_val : Validation data mask
    features : feature column names
    targets :target column names
    oos_preds_df : Out-of-sample predictions DataFrame
    models_final : Dictionary to store final models
    fold_results : List of Fold results
    stats_history : List to store statistics
    timing_stats : List to store timing information for NN models. So i can compare fast/best models ! *so we kann use the best for time horizon)
    train_end : date
    test_start : date
    test_end : date
    n_splits : Number of splits
    """
    from sklearn.preprocessing import StandardScaler

    # 1. Data Preprocessing
    X_tr_raw = df.loc[mask_tr, features]
    X_tr_winsor = train_dataPrep(X_tr_raw)
    X_val_raw = df.loc[mask_val, features]

    # GMM clustering features
    X_tr_winsor, X_val_enhanced = create_per_fold_features(
        X_tr_winsor, X_val_raw, gmm_components=3
    )
    X_val_raw = X_val_enhanced

    # Define models to train based on selection
    models_to_train = []
    if model_selection == 1 or model_selection == 5:  # RF or All
        models_to_train.append(
            ("RandomForest", src.model_utils.RANDOM_FOREST, X_tr_winsor)
        )

    if model_selection == 2 or model_selection == 5:  # LSTM or All
        # Prepare scaled data for NN models
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_winsor)
        X_val_scaled = scaler.transform(X_val_raw)
        models_to_train.append(("LSTM", src.model_utils.LSTM, X_tr_scaled))

    if model_selection == 3 or model_selection == 5:  # TCN or All
        # Prepare scaled data for NN models if not already done
        if not (model_selection == 2 or model_selection == 5):
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_winsor)
            X_val_scaled = scaler.transform(X_val_raw)
        models_to_train.append(("TCN", src.model_utils.TCN, X_tr_scaled))

    if model_selection == 4 or model_selection == 5:  # TFT or All
        # Prepare scaled data for NN models if not already done
        if not (model_selection == 2 or model_selection == 3 or model_selection == 5):
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_winsor)
            X_val_scaled = scaler.transform(X_val_raw)
        models_to_train.append(("TFT", src.model_utils.TFT, X_tr_scaled))

    # 2. Train selected models
    for model_name, model_type, X_train_data in models_to_train:
        print(f"Training {model_name}...")
        model_start_time = time.time()

        for target in targets:
            print(f"  Training for target: {target}")
            y_train = df.loc[mask_tr, target]
            y_val = df.loc[mask_val, target]

            #  timing for comparison
            target_start = time.time()

            # Use appropriate validation data (scaled for NN, raw for RF)
            if model_type == src.model_utils.RANDOM_FOREST:
                X_val_data = X_val_raw
            else:
                X_val_data = (
                    X_val_scaled
                    if "X_val_scaled" in locals()
                    else scaler.transform(X_val_raw)
                )

            train_and_evaluate_model(
                model_type=model_type,
                X_train=X_train_data,
                y_train=y_train,
                X_val=X_val_data,
                y_val=y_val,
                target=target,
                fold_idx=fold_idx,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                features=features,
                oos_preds_df=oos_preds_df,
                mask_val=mask_val,
                timing_stats=timing_stats,
                models_final=models_final,
                fold_results=fold_results,
                stats_history=stats_history,
                n_splits=n_splits,
            )

            # Record timing for this specific model and target
            # we need the timeing not only vor comparing models, also to select the best quality/time ration for differnt time horizon.abs
            # h1, m15 day
            target_time = time.time() - target_start
            timing_stats.append(
                {
                    "phase": phase_num,  # Will be set by calling function
                    "fold": fold_idx,
                    "model": model_name,
                    "target": target,
                    "time_seconds": target_time,
                    "timestamp": pd.Timestamp.now(),
                }
            )

        model_total_time = time.time() - model_start_time
        print(f"{model_name} training completed in {model_total_time:.2f} seconds")


def run_phase_training(
    phase_num: int,
    df_full: pd.DataFrame,
    start_date: str,
    end_date: str,
    features: list[str],
    targets: list[str],
    stats_history: list[dict],
    fold_results: list[dict],
    models_final: dict,
    oos_preds_df: pd.DataFrame,
    timing_stats: list[dict],
    n_splits: int,
    model_selection: int,
) -> pd.DataFrame:
    """
    Run complete training pipeline for the first phase (HyperTraining.)
            phase_num : Phase number (1 or 2)
            df_full : Full dataset
            features :  column names
            targets :  column names
            stats_history : Statistics history container
            fold_results : Fold results container
            models_final :Final models container
            oos_preds_df : Out-of-sample predictions
    Returns     oos_preds_df_phase:df =oof predictions
    """
    print(f"Starte Phase {phase_num} Training...")

    # split up data, so we only use the time frame that was calculaten für this fold.
    df_phase = df_full[
        (df_full.index.get_level_values("date") >= start_date)
        & (df_full.index.get_level_values("date") <= end_date)
    ]

    if df_phase.empty:
        print(f"No data for Phase {phase_num} found.")
        return pd.DataFrame()

    print(f"Daten für Phase {phase_num}: {len(df_phase):,} Zeilen")
    print(
        f"Zeitraum: {df_phase.index.get_level_values('date').min()} bis {df_phase.index.get_level_values('date').max()}"
    )

    splitter = PurgeTimeSeriesSplit()

    print(
        f"Use {n_splits} Folds with {src.model_utils.TRAIN_WINDOW_YEARS} Years Trainings-Fenster"
    )
    print(
        f"Test-Fenster: {src.model_utils.TEST_SIZE_DAYS} Tage, Purge: {src.model_utils.PURGE_SIZE_DAYS} Tage.je nach dem wie"
    )

    # Initialize out-of-sample predictions DataFrame
    oos_preds_df_phase = pd.DataFrame(index=df_phase.index)
    for target in targets:
        oos_preds_df_phase[f"actual_{target}"] = df_phase[target]

    # Training loop over all folds
    for fold_idx, mask_tr, mask_val, train_end, test_start, test_end in splitter.split(
        df_phase, n_splits
    ):
        fold_start = time.time()

        print(f"Verarbeite Fold {fold_idx + 1} von {n_splits}...")
        print(
            f"Trainings-Ende: {train_end.date()}, Test: {test_start.date()} bis {test_end.date()}"
        )

        # Train models for this fold
        train_models_fold(
            phase_num,
            fold_idx,
            df_phase,
            mask_tr,
            mask_val,
            features,
            targets,
            oos_preds_df_phase,
            models_final,
            fold_results,
            stats_history,
            timing_stats,
            train_end,
            test_start,
            test_end,
            n_splits,
            model_selection,
        )

        fold_time = time.time() - fold_start
        print(f"Fold {fold_idx + 1} ready in {fold_time:.1f} Seconds")

    print(f"Phase {phase_num} summary:")
    print_summary_report(stats_history, targets, n_splits)

    return oos_preds_df_phase


def save_phase_results(
    phase_num: int,
    oos_preds_df: pd.DataFrame,
    fold_results: list[dict],
    stats_history: list[dict],
    models_final: dict,
    timing_stats: list[dict],
) -> None:
    """
     # Save predictions vor better oof-process documentation and cotrolling.
    # in later productive use, needend for explanations.

        phase_num )
        oos_preds_df   Out-of-sample predictions DataFrame
        fold_results  List of fold results
        stats_history    List of statistics history
        models_final  Dictionary of final models

    """
    print(f"Speichere Ergebnisse für Phase {phase_num}...")

    # Save predictions vor better oof-process documentation and cotrolling.
    # in later productive use, needend for explainabelity.
    pred_cols = [c for c in oos_preds_df.columns if c.startswith("pred_")]
    if pred_cols:
        oos_preds_final = oos_preds_df[oos_preds_df[pred_cols].notna().any(axis=1)]
        oos_preds_final.to_hdf(
            f"results/oos_preds_phase{phase_num}.h5", key="predictions", mode="w"
        )
        print(
            f" saved prediction: results/oos_preds_phase{phase_num}.h5 ({len(oos_preds_final):,} Zeilen)"
        )

    # Save fold results
    if fold_results:
        df_fold_results = pd.DataFrame(fold_results)
        df_fold_results.to_csv(
            f"results/training_fold_results_phase{phase_num}.csv", index=False
        )
        print(
            f" Fold-Ergebnisse saved: results/training_fold_results_phase{phase_num}.csv"
        )

    # summary statistics also so for later controllablity and explainablity.
    if stats_history:
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
        summary_stats.to_csv(
            f"results/training_stats_summary_phase{phase_num}.csv", index=False
        )
        print(
            f"Statistik-Zusammenfassung gespeichert: results/training_stats_summary_phase{phase_num}.csv"
        )

    # Save timing stats for NN models
    if timing_stats:
        df_timing = pd.DataFrame(timing_stats)
        df_timing.to_csv(f"results/nn_training_times_phase{phase_num}.csv", index=False)
        print(
            f"NN Trainingszeiten gespeichert: results/nn_training_times_phase{phase_num}.csv"
        )

        # Save models
        if models_final:
            import joblib

            joblib.dump(models_final, f"models/models_final_phase{phase_num}.joblib")
            print(f"Modelle gespeichert: models/models_final_phase{phase_num}.joblib")


def compare_models(phases_data: dict) -> None:
    """
    Compare all trained models across phases and create a summary comparison.

    phases_data : Dictionary containing results from all phases
    """
    print("\n" + "____" * 25)
    print("MODEL COMPARISON ACROSS PHASES")
    print("=" * 30)

    all_stats = []

    # Collect statistics from all phases
    for phase_num, phase_data in phases_data.items():
        if phase_data["stats_history"]:
            df_stats = pd.DataFrame(phase_data["stats_history"])
            df_stats["Phase"] = phase_num
            all_stats.append(df_stats)

    if not all_stats:
        print("No model results available for comparison :( .")  # oh no !
        return

    # Combine all statistics
    combined_stats = pd.concat(all_stats, ignore_index=True)

    # Create comparison summary
    comparison = (
        combined_stats.groupby(["Phase", "Model", "Target"])
        .agg(
            {
                "RMSE": ["mean", "std", "min"],
                "MAE": ["mean", "std", "min"],
                "Sharpe": [
                    "mean",
                    "std",
                    "max",
                ],  # Max Sharpe for best performance. Later deflation Sharp !!
            }
        )
        .round(4)
    )

    # Flatten column names
    comparison.columns = [f"{col[0]}_{col[1]}" for col in comparison.columns]
    comparison = comparison.reset_index()

    # Save comparison results
    comparison.to_csv("results/model_comparison_summary.csv", index=False)
    print(f"Model comparison saved: results/model_comparison_summary.csv")

    # Print comparison summary
    print("\nMODEL PERFORMANCE COMPARISON:")
    print("-" * 30)

    for phase in sorted(combined_stats["Phase"].unique()):
        phase_data = comparison[comparison["Phase"] == phase]
        print(f"\nPhase {phase} Results:")
        print("___" * 10)

        for _, row in phase_data.iterrows():
            print(
                f"{row['Model']:10s} | {row['Target']:15s} | "
                f"Sharpe: {row['Sharpe_max']:+.3f} | "
                f"RMSE: {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}"
            )

    # Find best models per target and phase
    print("\nBEST MODELS PER TARGET AND PHASE:")
    print("___" * 30)

    for phase in sorted(combined_stats["Phase"].unique()):
        phase_stats = combined_stats[combined_stats["Phase"] == phase]
        print(f"\nPhase {phase}:")

        for target in phase_stats["Target"].unique():
            target_stats = phase_stats[phase_stats["Target"] == target]
            best_model = target_stats.loc[target_stats["Sharpe"].idxmax()]

            print(
                f"  {target:15s} → {best_model['Model']:10s} "
                f"(Sharpe: {best_model['Sharpe']:+.3f})"
            )

    print("\n" + "=" * 60)


def main() -> None:
    """
    Execute the complete ML training pipeline step by step. All function here or in file model_utils
    its possible to start the phases seperatly over commandline.
    After the third run with RF and TCN  (each 6 hourse with the actual Vol filter) i split them.

    Command line arguments:
    --phase1 : Run only Phase 1 training
    --phase2 : Run only Phase 2 training
    --model : Model selection (1=RF, 2=LSTM, 3 =TCN, 4=TFT, 5=All)
    """
    print("Starte komplettes Model Training...")
    print("----" * 10)

    #  command line arguments, so we can start Phase 2 also seperatly
    parser = argparse.ArgumentParser(description="Train models in phases")
    parser.add_argument("--phase1", action="store_true", help="Run only Phase 1")
    parser.add_argument("--phase2", action="store_true", help="Run only Phase 2")
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="Model to train: 1=RF, 2=LSTM, 3 =TCN, 4=TFT, 5=All (default)",
    )
    args = parser.parse_args()

    # 1. Load data
    print("\n Loading data. .. ... .... .....")
    try:
        df = pd.read_hdf(IMPORT_FILE, key="data").sort_index()
    except Exception as e:
        print(f" Error loading {IMPORT_FILE}: {e}")
        return

    # 2. Identify targets and features
    targets, features = identify_targets_and_features(df)
    print(f" Found  {len(features)} features and {len(targets)} targets")

    # stats_history = []  # Per-fold statistics
    # fold_results = []  # Detailed fold results
    # models_final = {}  # Final models from last fold

    # Set up out-of-sample predictions DataFrame
    oos_preds_df = pd.DataFrame(index=df.index)
    for target in targets:
        oos_preds_df[f"actual_{target}"] = df[target]

    # Initialize  result containers for both phases
    phases_data = {
        1: {
            "start_date": src.model_utils.PHASE_1_START,
            "end_date": src.model_utils.PHASE_1_END,
            "stats_history": [],
            "fold_results": [],
            "models_final": {},
            "oos_preds_df": pd.DataFrame(),
            "timing_stats": [],
        },
        2: {
            "start_date": src.model_utils.PHASE_2_START,
            "end_date": src.model_utils.PHASE_2_END,
            "stats_history": [],
            "fold_results": [],
            "models_final": {},
            "oos_preds_df": pd.DataFrame(),
            "timing_stats": [],
        },
    }

    # Run Phas 1 if not explicitly skipped, you can
    if not args.phase2:
        phase1_data = phases_data[1]
        phase1_data["oos_preds_df"] = run_phase_training(
            1,
            df,
            phase1_data["start_date"],
            phase1_data["end_date"],
            features,
            targets,
            phase1_data["stats_history"],
            phase1_data["fold_results"],
            phase1_data["models_final"],
            phase1_data["oos_preds_df"],
            phase1_data["timing_stats"],
            src.model_utils.PHASE_1_N_SPLITS,
            args.model,
        )

    # Run Phase 2 if not explicitly skipped and data available.abs..do phase 1 without 2 ??
    phase2_data = phases_data[2]
    df_phase2_check = df[
        (df.index.get_level_values("date") >= phase2_data["start_date"])
        & (df.index.get_level_values("date") <= phase2_data["end_date"])
    ]

    if not args.phase1 and not df_phase2_check.empty:
        phase2_data["oos_preds_df"] = run_phase_training(
            2,
            df,
            phase2_data["start_date"],
            phase2_data["end_date"],
            features,
            targets,
            phase2_data["stats_history"],
            phase2_data["fold_results"],
            phase2_data["models_final"],
            phase2_data["oos_preds_df"],
            phase2_data["timing_stats"],
            src.model_utils.PHASE_2_N_SPLITS,
            args.model,
        )
    elif not args.phase1:
        print("No Data for Phase 2 was found. Oh.....")

    # Save results for both phases
    print("\n Save results for both phases...")

    for phase_num in [
        1,
        2,
    ]:  # phase 3 should be analsis of result (time, target, every model seperatly.....)
        phase_data = phases_data[phase_num]
        if phase_data["stats_history"]:  # Only save if phase was run
            save_phase_results(
                phase_num,
                phase_data["oos_preds_df"],
                phase_data["fold_results"],
                phase_data["stats_history"],
                phase_data["models_final"],
                phase_data["timing_stats"],
            )

    # Compare all trained models
    compare_models(phases_data)

    print("Alle Phasen erfolgreich abgeschlossen!")
    print("_-_" * 25)


if __name__ == "__main__":
    main()

# %%
