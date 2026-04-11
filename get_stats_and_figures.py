#!/usr/bin/env python3


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["mean_dice", "dice_label1", "dice_label2", "mean_iou", "hd95"]
PRIMARY_METRIC = "mean_dice"
LOWER_IS_BETTER = {"hd95"}



def load_results(outputs_dir: Path) -> pd.DataFrame:
    return pd.read_csv(outputs_dir / "results.csv")


def load_epoch_csvs(outputs_dir: Path) -> dict[str, pd.DataFrame]:
    """Returns {experiment_name: DataFrame} for every epochs_*.csv found."""
    epoch_data = {}
    for path in sorted(outputs_dir.glob("epochs_*.csv")):
        exp_name = path.stem.removeprefix("epochs_")
        epoch_data[exp_name] = pd.read_csv(path)
    return epoch_data


def compute_stats(results: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame with mean and std per experiment for every metric."""
    grouped = results.groupby("experiment")
    rows = []
    for exp_name, group in grouped:
        row = {"experiment": exp_name}
        for metric in METRICS:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_std"] = group[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def sort_experiments(stats: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """Sorts experiments best-to-worst by the given metric mean."""
    ascending = metric in LOWER_IS_BETTER
    return stats.sort_values(metric + "_mean", ascending=ascending).reset_index(drop=True)


def print_stats(stats: pd.DataFrame) -> None:
    print("\n========== RESULTS SUMMARY ==========")
    for _, row in stats.iterrows():
        print("\n%s" % row["experiment"])
        for metric in METRICS:
            print("  %s: %.4f +/- %.4f" % (metric, row[metric + "_mean"], row[metric + "_std"]))


def plot_experiment_comparison(stats: pd.DataFrame, metric: str, out_dir: Path) -> None:
    sorted_stats = sort_experiments(stats, metric)
    exp_names = sorted_stats["experiment"].tolist()
    means = sorted_stats[metric + "_mean"].values
    stds = sorted_stats[metric + "_std"].values

    fig, ax = plt.subplots(figsize=(max(8, len(exp_names) * 1.4), 5))
    x = np.arange(len(exp_names))
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("Experiment comparison - %s (mean ± std)" % metric)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    path = out_dir / ("comparison_%s.png" % metric)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved %s" % path)


def plot_per_fold(results: pd.DataFrame, stats: pd.DataFrame, metric: str, out_dir: Path) -> None:
    sorted_stats = sort_experiments(stats, metric)
    exp_order = sorted_stats["experiment"].tolist()
    folds = sorted(results["fold"].unique())

    fig, axes = plt.subplots(1, len(folds), figsize=(4 * len(folds), 5), sharey=True)
    if len(folds) == 1:
        axes = [axes]

    for ax, fold in zip(axes, folds):
        fold_data = results[results["fold"] == fold].set_index("experiment")
        values = [fold_data.loc[e, metric] if e in fold_data.index else float("nan") for e in exp_order]
        x = np.arange(len(exp_order))
        ax.bar(x, values, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_order, rotation=30, ha="right", fontsize=7)
        ax.set_title("Fold %d" % fold)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel(metric)
    fig.suptitle("Per-fold results - %s" % metric, fontsize=18)
    fig.tight_layout()

    path = out_dir / ("per_fold_%s.png" % metric)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved %s" % path)


def plot_training_curves(epoch_data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    curves_dir = out_dir / "training_curves"
    curves_dir.mkdir(exist_ok=True)

    for exp_name, df in epoch_data.items():
        folds = sorted(df["fold"].unique())
        fig, axes = plt.subplots(1, len(folds), figsize=(5 * len(folds), 4), sharey=False)
        if len(folds) == 1:
            axes = [axes]

        for ax, fold in zip(axes, folds):
            fold_df = df[df["fold"] == fold].reset_index(drop=True)
            ax2 = ax.twinx()
            ax.plot(fold_df["epoch"], fold_df["train_loss"], color="blue", label="train_loss")
            ax2.plot(fold_df["epoch"], fold_df["val_mean_dice"], color="orange", label="val_mean_dice")
            ax.set_xlabel("epoch")
            ax.set_ylabel("train_loss", color="blue")
            ax2.set_ylabel("val_mean_dice", color="orange")
            ax.set_title("Fold %d" % fold)
            ax.grid(linestyle="--", alpha=0.4)

            # Combined legend
            lines = ax.get_lines() + ax2.get_lines()
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, fontsize=7)

        fig.suptitle("Training curves - %s" % exp_name, fontsize=18)
        fig.tight_layout()

        path = curves_dir / ("%s.png" % exp_name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("Saved %s" % path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stats and generate figures from experiment outputs")
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory containing results.csv and epochs_*.csv (default: outputs_1_10_10)",
    )
    parser.add_argument(
        "--metric",
        default=PRIMARY_METRIC,
        choices=METRICS,
        help="Primary metric used for ordering experiments (default: %s)" % PRIMARY_METRIC,
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        raise FileNotFoundError("Outputs directory not found: %s" % outputs_dir)

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    results = load_results(outputs_dir)
    epoch_data = load_epoch_csvs(outputs_dir)
    stats = compute_stats(results)

    print_stats(sort_experiments(stats, args.metric))

    plot_experiment_comparison(stats, args.metric, figures_dir)
    plot_per_fold(results, stats, args.metric, figures_dir)
    plot_training_curves(epoch_data, figures_dir)

if __name__ == "__main__":
    main()
