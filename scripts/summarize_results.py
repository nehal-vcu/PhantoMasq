"""Build the paper's tables from the per-capture results.

Per-type values average a type's capture instances for each (omega, delta).
The comparison table also needs final_consolidated_performance_metrics.csv from
the Marfo et al. repository, passed with --baseline_csv.
"""

import argparse
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ATTACK_TYPES = {
    "correlated_signal": "Correlated signal",
    "max_speedometer": "Max speedometer",
    "max_engine_coolant_temp": "Max engine coolant",
    "reverse_light_off": "Reverse lights off",
    "reverse_light_on": "Reverse lights on",
}

BASELINE_TYPES = {
    "correlated_signal_attack": "Correlated signal",
    "max_speedometer_attack": "Max speedometer",
    "max_engine_coolant_attack": "Max engine coolant",
    "reverse_light_off_attack": "Reverse lights off",
    "reverse_light_on_attack": "Reverse lights on",
}

# Reported by Marfo et al. for time-based windows on different hardware, so these
# are not directly comparable with PhantoMasq's TTW.
BASELINE_TIME_BASED_TTW_MS = {
    "Correlated signal": "24.00 - 6840.00",
    "Max speedometer": "5.00 - 20820.00",
    "Max engine coolant": "237.00 - 4890.00",
    "Reverse lights off": "27.00 - 6220.00",
    "Reverse lights on": "5.00 - 8530.00",
}


def attack_type(attack):
    return next(name for prefix, name in ATTACK_TYPES.items() if attack.startswith(prefix))


def load_results(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    results.insert(0, "attack_type", results.attack.map(attack_type))
    return results


def ttw_summary(results):
    rows = []
    for name in ATTACK_TYPES.values():
        subset = results[results.attack_type == name]
        fastest = subset.loc[subset.e2e_ttw_ms.idxmin()]
        slowest = subset.loc[subset.e2e_ttw_ms.idxmax()]
        rows.append({
            "Attack Type": name,
            "Mean": round(subset.e2e_ttw_ms.mean(), 2),
            "Std Dev": round(subset.e2e_ttw_ms.std(), 2),
            "Median": round(subset.e2e_ttw_ms.median(), 2),
            "Minimum": f"{fastest.e2e_ttw_ms:.2f} (omega={fastest.omega}, delta={fastest.delta})",
            "Maximum": f"{slowest.e2e_ttw_ms:.2f} (omega={slowest.omega}, delta={slowest.delta})",
        })
    return pd.DataFrame(rows)


def per_type_grid(results):
    return results.groupby(["attack_type", "omega", "delta"]).agg(
        auc_roc=("test_auc_mean", "mean"),
        auc_pr=("test_auc_pr_mean", "mean"),
        prevalence=("test_attack_ratio", "mean"),
        model_ttw=("model_side_ttw_ms", "mean"),
        e2e_ttw=("e2e_ttw_ms", "mean"),
        throughput=("e2e_throughput_windows_per_sec", "mean"),
    ).reset_index()


def best_configurations(grid):
    rows = []
    for name in ATTACK_TYPES.values():
        best = grid[grid.attack_type == name].nlargest(1, "auc_roc").iloc[0]
        rows.append({
            "Attack Type": name,
            "omega": int(best.omega),
            "delta": int(best.delta),
            "AUC-ROC": round(best.auc_roc, 3),
            "AUC-PR": round(best.auc_pr, 3),
            "Prevalence": round(best.prevalence, 3),
            "Model-Side TTW (ms)": round(best.model_ttw, 3),
            "End-to-End TTW (ms)": round(best.e2e_ttw, 2),
            "Throughput (win/s)": int(round(best.throughput)),
        })
    return pd.DataFrame(rows)


def baseline_comparison(grid, baseline_csv):
    baseline = pd.read_csv(baseline_csv)
    baseline = baseline[baseline.version == "all_normalized"].copy()
    baseline["attack_type"] = baseline.attack_type.map(BASELINE_TYPES)

    rows = []
    for name in ATTACK_TYPES.values():
        ours = grid[grid.attack_type == name]
        theirs = baseline[baseline.attack_type == name].roc_auc
        rows.append({
            "Attack Type": name,
            "PhantoMasq AUC-ROC": f"{ours.auc_roc.min():.3f} - {ours.auc_roc.max():.3f}",
            "Marfo AUC-ROC": f"{theirs.min():.2f} - {theirs.max():.2f}",
            "PhantoMasq AUC-PR": f"{ours.auc_pr.min():.3f} - {ours.auc_pr.max():.3f}",
            "PhantoMasq TTW": f"{ours.e2e_ttw.min():.2f} - {ours.e2e_ttw.max():.2f}",
            "Marfo TTW (time-based)": BASELINE_TIME_BASED_TTW_MS[name],
        })
    rows.append({
        "Attack Type": "Overall Range",
        "PhantoMasq AUC-ROC": f"{grid.auc_roc.min():.3f} - {grid.auc_roc.max():.3f}",
        "Marfo AUC-ROC": f"{baseline.roc_auc.min():.2f} - {baseline.roc_auc.max():.2f}",
        "PhantoMasq AUC-PR": f"{grid.auc_pr.min():.3f} - {grid.auc_pr.max():.3f}",
        "PhantoMasq TTW": f"{grid.e2e_ttw.min():.2f} - {grid.e2e_ttw.max():.2f}",
        "Marfo TTW (time-based)": "5.00 - 20820.00",
    })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default=os.path.join(ROOT, "results", "captures"))
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--baseline_csv", help="Marfo et al. consolidated metrics CSV")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    grid = per_type_grid(results)

    results.to_csv(os.path.join(args.output_dir, "all_results.csv"), index=False)
    ttw_summary(results).to_csv(os.path.join(args.output_dir, "table_ttw_summary.csv"), index=False)
    best_configurations(grid).to_csv(
        os.path.join(args.output_dir, "table_best_configurations.csv"), index=False)
    if args.baseline_csv:
        baseline_comparison(grid, args.baseline_csv).to_csv(
            os.path.join(args.output_dir, "table_baseline_comparison.csv"), index=False)
    print(f"wrote tables to {args.output_dir}")


if __name__ == "__main__":
    main()
