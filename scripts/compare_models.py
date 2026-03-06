"""
Model Comparison Script

Analyzes training results from all 4 models (RF, LSTM, TCN, TFT) and creates comparative visualizations.
Compares performance metrics and training times across models and phases.

Usage:
    python scripts/compare_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")


def load_training_results(phase_num: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training results and timing data for a specific phase.

    Args:
        phase_num: Phase number (1 or 2)

    Returns:
        Tuple of (stats_df, timing_df)
    """
    base_path = Path("results")

    # Load training statistics
    stats_file = base_path / f"training_stats_summary_phase{phase_num}.csv"
    if not stats_file.exists():
        print(f"Warning: {stats_file} not found")
        stats_df = pd.DataFrame()
    else:
        stats_df = pd.read_csv(stats_file)
        stats_df["Phase"] = phase_num

    # Load timing statistics
    timing_file = base_path / f"nn_training_times_phase{phase_num}.csv"
    if not timing_file.exists():
        print(f"Warning: {timing_file} not found")
        timing_df = pd.DataFrame()
    else:
        timing_df = pd.read_csv(timing_file)
        timing_df["Phase"] = phase_num

    return stats_df, timing_df


def create_performance_comparison(stats_df: pd.DataFrame, phase_num: int) -> None:
    """
    Create performance comparison plots for a single phase.

    Args:
        stats_df: Training statistics DataFrame
        phase_num: Phase number
    """
    if stats_df.empty:
        print(f"No statistics data for Phase {phase_num}")
        return

    # Set up the plotting area
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Model Performance Comparison - Phase {phase_num}",
        fontsize=16,
        fontweight="bold",
    )

    # Colors for different models
    model_colors = {
        "RandomForest": "#1f77b4",  # Blue
        "LSTM": "#ff7f0e",  # Orange
        "TCN": "#2ca02c",  # Green
        "TFT": "#d62728",  # Red
    }

    # 1. Sharpe Ratio Comparison
    ax1 = axes[0, 0]
    sharpe_data = stats_df.pivot(index="Target", columns="Model", values="Sharpe_mean")
    sharpe_data.plot(
        kind="bar",
        ax=ax1,
        color=[model_colors.get(m, "gray") for m in sharpe_data.columns],
    )
    ax1.set_title("Average Sharpe Ratio by Model and Target")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.tick_params(axis="x", rotation=45)

    # 2. RMSE Comparison
    ax2 = axes[0, 1]
    rmse_data = stats_df.pivot(index="Target", columns="Model", values="RMSE_mean")
    rmse_data.plot(
        kind="bar",
        ax=ax2,
        color=[model_colors.get(m, "gray") for m in rmse_data.columns],
    )
    ax2.set_title("Average RMSE by Model and Target")
    ax2.set_ylabel("RMSE")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.tick_params(axis="x", rotation=45)

    # 3. Placeholder for additional metrics
    ax3 = axes[1, 0]
    ax3.text(
        0.5,
        0.5,
        "Additional metrics\ncan be added here",
        ha="center",
        va="center",
        transform=ax3.transAxes,
    )
    ax3.set_title("Additional Analysis")

    # 4. Model Ranking Summary
    ax4 = axes[1, 1]

    # Calculate average performance metrics
    summary_stats = (
        stats_df.groupby("Model")
        .agg({"Sharpe_mean": "mean", "RMSE_mean": "mean", "MAE_mean": "mean"})
        .round(4)
    )

    # Normalize metrics (higher Sharpe and lower RMSE/MAE is better)
    summary_stats["Sharpe_norm"] = (
        summary_stats["Sharpe_mean"] - summary_stats["Sharpe_mean"].min()
    ) / (summary_stats["Sharpe_mean"].max() - summary_stats["Sharpe_mean"].min())

    summary_stats["RMSE_norm"] = 1 - (
        summary_stats["RMSE_mean"] - summary_stats["RMSE_mean"].min()
    ) / (summary_stats["RMSE_mean"].max() - summary_stats["RMSE_mean"].min())

    summary_stats["MAE_norm"] = 1 - (
        summary_stats["MAE_mean"] - summary_stats["MAE_mean"].min()
    ) / (summary_stats["MAE_mean"].max() - summary_stats["MAE_mean"].min())

    # Overall score
    summary_stats["Overall_Score"] = (
        summary_stats["Sharpe_norm"]
        + summary_stats["RMSE_norm"]
        + summary_stats["MAE_norm"]
    ) / 3

    summary_stats["Overall_Score"].sort_values(ascending=True).plot(
        kind="barh",
        ax=ax4,
        color=[model_colors.get(m, "gray") for m in summary_stats.index],
    )
    ax4.set_title("Overall Performance Score (Higher is Better)")
    ax4.set_xlabel("Normalized Score")

    plt.tight_layout()
    plt.savefig(
        f"reports/figures/model_comparison_phase{phase_num}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Print summary table
    print(f"\n=== Phase {phase_num} Performance Summary ===")
    print(
        summary_stats[["Sharpe_mean", "RMSE_mean", "MAE_mean", "Overall_Score"]].round(
            4
        )
    )
    print()


def create_cross_phase_comparison(
    all_stats: pd.DataFrame, all_timing: pd.DataFrame
) -> None:
    """
    Create comparison plots across both phases.

    Args:
        all_stats: Combined statistics from both phases
        all_timing: Combined timing data from both phases
    """
    if all_stats.empty:
        print("No cross-phase comparison data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Cross-Phase Model Comparison", fontsize=16, fontweight="bold")

    # Colors for phases
    phase_colors = {1: "#1f77b4", 2: "#ff7f0e"}  # Blue for Phase 1, Orange for Phase 2

    # 1. Sharpe Ratio by Phase and Model
    ax1 = axes[0, 0]
    sharpe_pivot = all_stats.pivot_table(
        values="Sharpe_mean", index="Model", columns="Phase", aggfunc="mean"
    )
    sharpe_pivot.plot(
        kind="bar", ax=ax1, color=[phase_colors[p] for p in sharpe_pivot.columns]
    )
    ax1.set_title("Average Sharpe Ratio by Model and Phase")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend(title="Phase", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.tick_params(axis="x", rotation=45)

    # 2. Training Time Comparison by Phase
    ax2 = axes[0, 1]
    if not all_timing.empty:
        time_by_phase_model = all_timing.pivot_table(
            values="time_seconds", index="model", columns="phase", aggfunc="mean"
        )
        time_by_phase_model.plot(
            kind="bar",
            ax=ax2,
            color=[phase_colors[p] for p in time_by_phase_model.columns],
        )
        ax2.set_title("Average Training Time by Model and Phase")
        ax2.set_ylabel("Time (seconds)")
        ax2.legend(title="Phase", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.tick_params(axis="x", rotation=45)
    else:
        ax2.text(
            0.5,
            0.5,
            "No timing data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Training Time Comparison by Phase")

    # 3. Performance Stability (Sharpe Std)
    ax3 = axes[1, 0]
    stability_data = all_stats.pivot_table(
        values="Sharpe_std", index="Model", columns="Phase", aggfunc="mean"
    )
    stability_data.plot(
        kind="bar", ax=ax3, color=[phase_colors[p] for p in stability_data.columns]
    )
    ax3.set_title("Performance Stability (Sharpe Ratio Std Dev)")
    ax3.set_ylabel("Sharpe Ratio Std Dev (Lower is Better)")
    ax3.legend(title="Phase", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.tick_params(axis="x", rotation=45)

    # 4. Overall Ranking
    ax4 = axes[1, 1]

    # Calculate overall scores by phase
    phase_summaries = {}
    for phase in [1, 2]:
        phase_data = all_stats[all_stats["Phase"] == phase]
        if not phase_data.empty:
            summary_stats = phase_data.groupby("Model").agg(
                {"Sharpe_mean": "mean", "RMSE_mean": "mean", "MAE_mean": "mean"}
            )

            # Normalize metrics
            summary_stats["Sharpe_norm"] = (
                summary_stats["Sharpe_mean"] - summary_stats["Sharpe_mean"].min()
            ) / (
                summary_stats["Sharpe_mean"].max() - summary_stats["Sharpe_mean"].min()
            )

            summary_stats["RMSE_norm"] = 1 - (
                summary_stats["RMSE_mean"] - summary_stats["RMSE_mean"].min()
            ) / (summary_stats["RMSE_mean"].max() - summary_stats["RMSE_mean"].min())

            summary_stats["MAE_norm"] = 1 - (
                summary_stats["MAE_mean"] - summary_stats["MAE_mean"].min()
            ) / (summary_stats["MAE_mean"].max() - summary_stats["MAE_mean"].min())

            summary_stats["Overall_Score"] = (
                summary_stats["Sharpe_norm"]
                + summary_stats["RMSE_norm"]
                + summary_stats["MAE_norm"]
            ) / 3
            phase_summaries[phase] = summary_stats["Overall_Score"]

    if phase_summaries:
        ranking_df = pd.DataFrame(phase_summaries)
        ranking_df = ranking_df.sort_values(
            by=list(ranking_df.columns), ascending=False
        )
        ranking_df.plot(
            kind="bar", ax=ax4, color=[phase_colors[p] for p in ranking_df.columns]
        )
        ax4.set_title("Overall Performance Ranking by Phase")
        ax4.set_ylabel("Overall Score")
        ax4.legend(title="Phase", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        "reports/figures/cross_phase_model_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def create_nn_models_comparison(timing_df: pd.DataFrame) -> None:
    """
    Create detailed comparison of NN models only.

    Args:
        timing_df: Combined timing data from both phases
    """
    if timing_df.empty:
        print("No NN timing data available for comparison")
        return

    # Filter for NN models only
    nn_models = ["LSTM", "TCN", "TFT"]
    nn_timing = timing_df[timing_df["model"].isin(nn_models)]

    if nn_timing.empty:
        print("No NN model timing data found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Neural Network Models Detailed Comparison", fontsize=16, fontweight="bold"
    )

    # Colors for NN models
    nn_colors = {
        "LSTM": "#ff7f0e",  # Orange
        "TCN": "#2ca02c",  # Green
        "TFT": "#d62728",  # Red
    }

    # 1. Training Time Distribution
    ax1 = axes[0, 0]
    sns.boxplot(data=nn_timing, x="model", y="time_seconds", ax=ax1, palette=nn_colors)
    ax1.set_title("Training Time Distribution by NN Model")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_xlabel("Model")

    # 2. Training Time by Phase
    ax2 = axes[0, 1]
    time_by_phase = nn_timing.pivot_table(
        values="time_seconds", index="model", columns="phase", aggfunc="mean"
    )
    time_by_phase.plot(
        kind="bar",
        ax=ax2,
        color=[nn_colors.get(m, "gray") for m in time_by_phase.index],
    )
    ax2.set_title("Average Training Time by Model and Phase")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_xlabel("Model")
    ax2.tick_params(axis="x", rotation=45)

    # 3. Time vs Target Analysis
    ax3 = axes[1, 0]
    target_time = (
        nn_timing.groupby(["model", "target"])["time_seconds"].mean().reset_index()
    )
    sns.scatterplot(
        data=target_time,
        x="target",
        y="time_seconds",
        hue="model",
        ax=ax3,
        palette=nn_colors,
        s=100,
    )
    ax3.set_title("Training Time by Target and Model")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_xlabel("Target")
    ax3.tick_params(axis="x", rotation=45)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 4. Performance vs Speed Trade-off (if performance data available)
    ax4 = axes[1, 1]
    # This would require combining timing with performance data
    ax4.text(
        0.5,
        0.5,
        "Performance vs Speed analysis\nwould require combined datasets",
        ha="center",
        va="center",
        transform=ax4.transAxes,
        fontsize=12,
    )
    ax4.set_title("Performance vs Speed Trade-off")

    plt.tight_layout()
    plt.savefig(
        "reports/figures/nn_models_detailed_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Print NN models summary
    print("\n=== Neural Network Models Summary ===")
    nn_summary = (
        nn_timing.groupby("model")
        .agg({"time_seconds": ["mean", "std", "min", "max", "count"]})
        .round(2)
    )
    nn_summary.columns = ["_".join(col).strip() for col in nn_summary.columns.values]
    print(nn_summary)


def generate_report(all_stats: pd.DataFrame, all_timing: pd.DataFrame) -> None:
    """
    Generate a comprehensive text report.

    Args:
        all_stats: Combined statistics from both phases
        all_timing: Combined timing data from both phases
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)

    print("\n1. OVERVIEW")
    print(
        f"Total models compared: {len(all_stats['Model'].unique()) if not all_stats.empty else 0}"
    )
    print(
        f"Phases analyzed: {sorted(all_stats['Phase'].unique()) if not all_stats.empty else 'None'}"
    )
    print(
        f"Targets analyzed: {len(all_stats['Target'].unique()) if not all_stats.empty else 0}"
    )

    if not all_stats.empty:
        print("\n2. PERFORMANCE SUMMARY BY MODEL")

        # Overall performance ranking
        overall_perf = (
            all_stats.groupby("Model")
            .agg({"Sharpe_mean": "mean", "RMSE_mean": "mean", "MAE_mean": "mean"})
            .round(4)
        )

        print("\nAverage Performance Metrics:")
        print(overall_perf.to_string())

        # Best model identification
        best_sharpe = overall_perf["Sharpe_mean"].idxmax()
        best_rmse = overall_perf["RMSE_mean"].idxmin()

        print(
            f"\nBest Sharpe Ratio: {best_sharpe} ({overall_perf.loc[best_sharpe, 'Sharpe_mean']:.4f})"
        )
        print(
            f"Best RMSE: {best_rmse} ({overall_perf.loc[best_rmse, 'RMSE_mean']:.4f})"
        )

    if not all_timing.empty:
        print("\n3. TRAINING TIME ANALYSIS")

        time_summary = (
            all_timing.groupby("model")
            .agg({"time_seconds": ["mean", "std", "sum"]})
            .round(2)
        )

        print("\nTraining Time Summary:")
        print(time_summary.to_string())

        fastest_model = time_summary[("time_seconds", "mean")].idxmin()
        slowest_model = time_summary[("time_seconds", "mean")].idxmax()

        print(
            f"\nFastest model: {fastest_model} ({time_summary.loc[fastest_model, ('time_seconds', 'mean')]:.2f}s avg)"
        )
        print(
            f"Slowest model: {slowest_model} ({time_summary.loc[slowest_model, ('time_seconds', 'mean')]:.2f}s avg)"
        )

    print("\n4. NEURAL NETWORK MODELS COMPARISON")

    nn_models = ["LSTM", "TCN", "TFT"]
    nn_stats = (
        all_stats[all_stats["Model"].isin(nn_models)]
        if not all_stats.empty
        else pd.DataFrame()
    )
    nn_timing = (
        all_timing[all_timing["model"].isin(nn_models)]
        if not all_timing.empty
        else pd.DataFrame()
    )

    if not nn_stats.empty:
        nn_perf = (
            nn_stats.groupby("Model")
            .agg({"Sharpe_mean": "mean", "RMSE_mean": "mean"})
            .round(4)
        )

        print("\nNN Models Performance:")
        print(nn_perf.to_string())

    if not nn_timing.empty:
        nn_time = nn_timing.groupby("model").agg({"time_seconds": "mean"}).round(2)

        print("\nNN Models Training Time:")
        print(nn_time.to_string())

        # Combined performance/time analysis
        if not nn_perf.empty:
            print("\nNN Models Efficiency (Sharpe per second):")
            efficiency = nn_perf["Sharpe_mean"] / nn_time["time_seconds"]
            efficiency = efficiency.sort_values(ascending=False)
            for model, score in efficiency.items():
                print(f"{model}: {score:.6f}")

    print("\n5. RECOMMENDATIONS")

    if not all_stats.empty and not all_timing.empty:
        # Simple recommendation logic
        best_overall = overall_perf["Sharpe_mean"].idxmax()
        fastest = time_summary[("time_seconds", "mean")].idxmin()

        print(f"- Best overall performance: {best_overall}")
        print(f"- Fastest training: {fastest}")

        if best_overall in nn_models:
            print(f"- For NN models, consider {best_overall} for best performance")
            nn_best = nn_perf["Sharpe_mean"].idxmax()
            nn_fastest = nn_time["time_seconds"].idxmin()
            if nn_best != nn_fastest:
                print(
                    f"- Trade-off: {nn_best} performs better, {nn_fastest} trains faster"
                )

    print("\n" + "=" * 80)
    print("Report generated. Visualizations saved to reports/figures/")
    print("=" * 80)


def main():
    """Main comparison function."""
    print("Starting Model Comparison Analysis...")

    # Load data from both phases
    all_stats = []
    all_timing = []

    for phase in [1, 2]:
        stats_df, timing_df = load_training_results(phase)

        if not stats_df.empty:
            all_stats.append(stats_df)
        if not timing_df.empty:
            all_timing.append(timing_df)

    # Combine data
    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)
    else:
        combined_stats = pd.DataFrame()

    if all_timing:
        combined_timing = pd.concat(all_timing, ignore_index=True)
    else:
        combined_timing = pd.DataFrame()

    # Create visualizations
    if not combined_stats.empty:
        for phase in [1, 2]:
            phase_stats = combined_stats[combined_stats["Phase"] == phase]
            if not phase_stats.empty:
                create_performance_comparison(phase_stats, phase)

        # Cross-phase comparison
        create_cross_phase_comparison(combined_stats, combined_timing)

    # NN models specific comparison
    if not combined_timing.empty:
        create_nn_models_comparison(combined_timing)

    # Generate text report
    generate_report(combined_stats, combined_timing)

    print("\nModel comparison completed!")
    print("Check reports/figures/ for generated plots.")


if __name__ == "__main__":
    main()
