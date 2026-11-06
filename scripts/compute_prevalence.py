"""Compute attack-window prevalence for every capture and (omega, delta) setting.

Prevalence is the fraction of windows labeled malicious. It depends only on
windowing and labeling, so no training is needed.
"""

import argparse
import csv
import os

from phantomasq.data import (
    load_injection_interval,
    make_can_df,
    partition_attack_regions,
    road_paths,
    temporal_split,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def n_full_windows(n_frames, omega, delta):
    """Number of windows of exactly omega frames in a region of n_frames."""
    return (n_frames - omega) // delta + 1 if n_frames >= omega else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road_dir", required=True, help="Root of the ROAD dataset")
    parser.add_argument("--grid", default=os.path.join(ROOT, "configs", "grid.csv"))
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "prevalence.csv"))
    args = parser.parse_args()

    with open(args.grid, newline="") as f:
        grid = list(csv.DictReader(f))

    ambient = grid[0]["ambient"]
    paths = road_paths(args.road_dir, grid[0]["attack"], ambient)
    ambient_split = temporal_split(make_can_df(paths["ambient_log"]))

    regions_by_attack = {}
    for attack in dict.fromkeys(config["attack"] for config in grid):
        paths = road_paths(args.road_dir, attack, ambient)
        inj_start, inj_end = load_injection_interval(paths["metadata"], attack)
        before, attack_region, after = partition_attack_regions(
            make_can_df(paths["attack_log"]), inj_start, inj_end)
        regions_by_attack[attack] = {
            "attack": temporal_split(attack_region),
            "benign": [ambient_split, temporal_split(before), temporal_split(after)],
        }
        print(f"parsed {attack}", flush=True)

    rows = []
    for config in grid:
        attack = config["attack"]
        omega, delta = int(config["window_size"]), int(config["window_offset"])
        regions = regions_by_attack[attack]
        record = {"attack": attack, "omega": omega, "delta": delta}
        for split_index, split_name in enumerate(("train", "test")):
            positives = n_full_windows(len(regions["attack"][split_index]), omega, delta)
            negatives = sum(n_full_windows(len(region[split_index]), omega, delta)
                            for region in regions["benign"])
            record[f"{split_name}_positives"] = positives
            record[f"{split_name}_windows"] = positives + negatives
            record[f"{split_name}_prevalence"] = round(positives / (positives + negatives), 6)
        rows.append(record)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
