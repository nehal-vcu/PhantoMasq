"""Run every experiment in a grid file and collect one results CSV per capture.

A failed run is recorded with status=failed and the sweep continues. Completed
configurations are skipped, so re-running the same command resumes a sweep.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELDS = [
    "attack", "omega", "delta",
    "test_auc_mean", "test_auc_std", "test_auc_per_seed",
    "test_auc_pr_mean", "test_auc_pr_std", "test_auc_pr_per_seed", "n_seeds",
    "train_graphs", "train_positives", "train_attack_ratio",
    "test_graphs", "test_positives", "test_attack_ratio",
    "model_side_ttw_ms", "throughput_windows_per_sec", "num_test_windows",
    "e2e_ttw_ms", "e2e_throughput_windows_per_sec", "e2e_num_windows",
    "ttw_device", "epochs", "status", "timestamp",
]


def row_key(row):
    return str(row["omega"]), str(row["delta"])


def load_rows(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {row_key(row): row for row in csv.DictReader(f)}


def save_rows(path, rows):
    ordered = sorted(rows.values(), key=lambda r: (int(r["omega"]), int(r["delta"])))
    temporary = path + ".tmp"
    with open(temporary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    shutil.move(temporary, path)


def run_step(command, log_path):
    with open(log_path, "w") as log:
        return subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road_dir", required=True, help="Root of the ROAD dataset")
    parser.add_argument("--captures_dir", required=True, help="Directory of ACTT capture folders")
    parser.add_argument("--actt_src", required=True, help="ACTT src directory")
    parser.add_argument("--grid", default=os.path.join(ROOT, "configs", "grid.csv"))
    parser.add_argument("--results_dir", default=os.path.join(ROOT, "results", "rerun"))
    parser.add_argument("--runs_dir", default=os.path.join(ROOT, "runs"),
                        help="Per-run logs, JSON files and checkpoints")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    with open(args.grid, newline="") as f:
        grid = list(csv.DictReader(f))

    results_dir = os.path.abspath(args.results_dir)
    runs_dir = os.path.abspath(args.runs_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    shared = ["--road_dir", os.path.abspath(args.road_dir),
              "--captures_dir", os.path.abspath(args.captures_dir),
              "--actt_src", os.path.abspath(args.actt_src)]

    for i, config in enumerate(grid, start=1):
        attack = config["attack"]
        omega, delta = config["window_size"], config["window_offset"]
        results_path = os.path.join(results_dir, f"{attack}.csv")
        rows = load_rows(results_path)
        progress = f"[{i}/{len(grid)}] {attack} omega={omega} delta={delta}"

        if (omega, delta) in rows:
            print(f"{progress}: already done")
            continue

        run_dir = os.path.join(runs_dir, f"{attack}__w{omega}_d{delta}")
        os.makedirs(run_dir, exist_ok=True)
        train_json = os.path.join(run_dir, "train_eval.json")
        ttw_json = os.path.join(run_dir, "measure_ttw.json")
        model_path = os.path.join(run_dir, "model.pt")
        window = ["--window_size", omega, "--window_offset", delta]
        capture = ["--attack", attack, "--attack_capture", config["attack_capture"]]

        train_command = [
            sys.executable, "-m", "scripts.train_eval", *shared, *capture, *window,
            "--ambient", config["ambient"], "--ambient_capture", config["ambient_capture"],
            "--seeds", args.seeds, "--output_json", train_json, "--save_model", model_path,
        ]
        ttw_command = [
            sys.executable, "-m", "scripts.measure_ttw", *shared, *capture, *window,
            "--model_path", model_path, "--output_json", ttw_json,
        ]

        if args.dry_run:
            print(f"{progress}\n  {' '.join(train_command)}\n  {' '.join(ttw_command)}")
            continue

        print(f"{progress}: training", flush=True)
        started = time.time()
        row = {"attack": attack, "omega": omega, "delta": delta,
               "timestamp": datetime.now().isoformat(timespec="seconds"), "status": "failed"}

        if run_step(train_command, os.path.join(run_dir, "train_eval.log")) == 0:
            with open(train_json) as f:
                train = json.load(f)
            splits = train["splits"]
            row.update({
                "test_auc_mean": round(train["test_auc_mean"], 6),
                "test_auc_std": round(train["test_auc_std"], 6),
                "test_auc_per_seed": ";".join(f"{v:.6f}" for v in train["test_auc_per_seed"]),
                "test_auc_pr_mean": round(train["test_auc_pr_mean"], 6),
                "test_auc_pr_std": round(train["test_auc_pr_std"], 6),
                "test_auc_pr_per_seed": ";".join(f"{v:.6f}" for v in train["test_auc_pr_per_seed"]),
                "n_seeds": len(train["seeds"]),
                "train_graphs": splits["train"]["graphs"],
                "train_positives": splits["train"]["positives"],
                "train_attack_ratio": round(splits["train"]["attack_ratio"], 4),
                "test_graphs": splits["test"]["graphs"],
                "test_positives": splits["test"]["positives"],
                "test_attack_ratio": round(splits["test"]["attack_ratio"], 4),
                "model_side_ttw_ms": round(train["model_side_ttw_ms"], 4),
                "throughput_windows_per_sec": round(train["throughput_windows_per_sec"], 2),
                "num_test_windows": train["num_test_windows"],
                "ttw_device": train["ttw_device"],
                "epochs": train["epochs"],
            })
            if run_step(ttw_command, os.path.join(run_dir, "measure_ttw.log")) == 0:
                with open(ttw_json) as f:
                    ttw = json.load(f)
                row.update({
                    "e2e_ttw_ms": round(ttw["ttw_ms"], 4),
                    "e2e_throughput_windows_per_sec": round(ttw["throughput_windows_per_sec"], 2),
                    "e2e_num_windows": ttw["num_windows"],
                    "status": "ok",
                })

        rows[(omega, delta)] = row
        save_rows(results_path, rows)
        print(f"  {row['status']} in {time.time() - started:.0f}s, AUC-ROC "
              f"{row.get('test_auc_mean', 'n/a')}, see {run_dir}", flush=True)


if __name__ == "__main__":
    main()
