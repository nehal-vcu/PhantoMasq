"""Draw the AUC-ROC and AUC-PR heatmaps across (omega, delta) for each attack type.

Run scripts.summarize_results first. Values average each type's capture instances.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANELS = [
    ("Correlated signal", "Average Correlated Signal Attack"),
    ("Max speedometer", "Average Max Speedometer Attack"),
    ("Max engine coolant", "Max Engine Coolant Attack"),
    ("Reverse lights off", "Average Reverse Lights Off Attack"),
    ("Reverse lights on", "Average Reverse Lights On Attack"),
]
WINDOWS = list(range(50, 401, 50))


def panel_grid(results, attack_type, column):
    """Return values indexed as [delta, omega], with NaN where delta > omega."""
    table = results[results.attack_type == attack_type].pivot_table(
        index="delta", columns="omega", values=column, aggfunc="mean")
    return table.reindex(index=WINDOWS, columns=WINDOWS).to_numpy(dtype=float)


def draw(results, column, label, output_path):
    grids = {attack_type: panel_grid(results, attack_type, column) for attack_type, _ in PANELS}
    smallest = min(np.nanmin(grid) for grid in grids.values())
    vmin, vmax = np.floor(smallest * 20) / 20, 1.0

    figure = plt.figure(figsize=(16.5, 9.2))
    layout = figure.add_gridspec(2, 6, hspace=0.35, wspace=1.1)
    axes = [figure.add_subplot(layout[0, 0:2]), figure.add_subplot(layout[0, 2:4]),
            figure.add_subplot(layout[0, 4:6]), figure.add_subplot(layout[1, 1:3]),
            figure.add_subplot(layout[1, 3:5])]

    for axis, (attack_type, title) in zip(axes, PANELS):
        grid = grids[attack_type]
        image = axis.imshow(grid, origin="lower", cmap="Reds", vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_xticks(range(len(WINDOWS)), WINDOWS, fontsize=9)
        axis.set_yticks(range(len(WINDOWS)), WINDOWS, fontsize=9)
        axis.set_xlabel("Window", fontsize=10)
        axis.set_ylabel("Offset", fontsize=10)
        axis.set_title(title, fontsize=11)
        for row, column_index in zip(*np.where(~np.isnan(grid))):
            value = grid[row, column_index]
            axis.text(column_index, row, f"{value:.3f}", ha="center", va="center", fontsize=7,
                      color="white" if value > (vmin + vmax) / 2 else "black")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=os.path.join(ROOT, "results", "all_results.csv"))
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "results", "figures"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = pd.read_csv(args.results)
    draw(results, "test_auc_mean", "Avg AUC_ROC", os.path.join(args.output_dir, "auc_roc.png"))
    draw(results, "test_auc_pr_mean", "Avg AUC_PR", os.path.join(args.output_dir, "auc_pr.png"))


if __name__ == "__main__":
    main()
