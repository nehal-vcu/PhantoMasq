"""Create the ACTT capture.pkl files that the experiments read, from the ROAD logs.

Runs ACTT's init_cancapture_from_canlog.py once for each capture named in the
grid file and moves the result into --captures_dir. Three details make the
output match the captures used in the paper:

- ACTT's parser always skips a log's first and last line, but ROAD logs have
  no header or footer. Each log is copied with one placeholder line at each end
  so that no frame is lost.
- ACTT names each capture folder after the log's start time in the local time
  zone. The paper's captures were made in US Eastern time, so ACTT runs with
  TZ=America/New_York.
- ACTT looks each capture name up as a VIN in the NHTSA vPIC web service. ROAD
  names are not VINs, so the lookup is replaced by a stub that reports no match.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs ACTT step 1 with the VIN lookup stubbed out. argv: actt_src, then ACTT's own arguments.
BOOTSTRAP = """
import runpy, sys, types
actt_src = sys.argv.pop(1)
sys.path.insert(0, actt_src)
no_match = types.SimpleNamespace(lookup_vin=lambda vin: types.SimpleNamespace(error_code=["1"]))
sys.modules["vpic"] = types.SimpleNamespace(vpic=no_match)
import CAN_objects.vehicle
CAN_objects.vehicle.vpic = no_match
script = f"{actt_src}/init_cancapture_from_canlog.py"
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
"""

# Capture folders are named <name>_<YYMMDD>_<HHMMSS>.
TIMESTAMP_SUFFIX = len("_YYMMDD_HHMMSS")


def captures_in_grid(grid_path, road_dir):
    """Return {capture folder name: ROAD log path} for every capture the grid uses."""
    captures = {}
    with open(grid_path, newline="") as f:
        for row in csv.DictReader(f):
            captures[row["attack_capture"]] = os.path.join(road_dir, "attacks", f"{row['attack']}.log")
            captures[row["ambient_capture"]] = os.path.join(road_dir, "ambient", f"{row['ambient']}.log")
    return captures


def make_capture(name, log_path, actt_src, captures_dir):
    with tempfile.TemporaryDirectory() as work:
        metadata = os.path.join(work, "metadata")
        os.makedirs(metadata)
        shutil.copy(os.path.join(actt_src, os.pardir, "metadata", "obd2_std_PIDs_processed.json"),
                    metadata)

        padded = os.path.join(work, os.path.basename(log_path))
        with open(log_path) as src, open(padded, "w") as dst:
            dst.write("placeholder header\n")
            shutil.copyfileobj(src, dst)
            dst.write("placeholder footer\n")

        env = dict(os.environ, TZ="America/New_York")
        command = [sys.executable, "-c", BOOTSTRAP, actt_src, padded,
                   "-v", name[:-TIMESTAMP_SUFFIX], "--silent"]
        subprocess.run(command, cwd=work, env=env, check=True)

        made = os.path.join(work, "data-cancaptures", name, "capture.pkl")
        if not os.path.exists(made):
            produced = os.listdir(os.path.join(work, "data-cancaptures"))
            raise SystemExit(f"Expected {name}, but ACTT produced {produced}")
        os.makedirs(os.path.join(captures_dir, name), exist_ok=True)
        shutil.move(made, os.path.join(captures_dir, name, "capture.pkl"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--road_dir", required=True, help="Root of the ROAD dataset")
    parser.add_argument("--actt_src", required=True, help="ACTT src directory")
    parser.add_argument("--captures_dir", required=True, help="Where to write the capture folders")
    parser.add_argument("--grid", default=os.path.join(ROOT, "configs", "grid.csv"))
    args = parser.parse_args()

    actt_src = os.path.abspath(args.actt_src)
    captures_dir = os.path.abspath(args.captures_dir)
    captures = captures_in_grid(args.grid, os.path.abspath(args.road_dir))

    for i, (name, log_path) in enumerate(sorted(captures.items()), start=1):
        progress = f"[{i}/{len(captures)}] {name}"
        if os.path.exists(os.path.join(captures_dir, name, "capture.pkl")):
            print(f"{progress}: already exists")
            continue
        print(f"{progress}: creating from {os.path.basename(log_path)}", flush=True)
        make_capture(name, log_path, actt_src, captures_dir)


if __name__ == "__main__":
    main()
