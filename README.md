# PhantoMasq

Code and results for the paper *Phantom of the Masquerade: Graph Neural Network-Based
Detection of Masquerade Attacks in Controller Area Networks*, by Nehal Ameen, Pablo
Moriano, and Irfan Ahmed.

PhantoMasq detects masquerade attacks on the CAN bus. It builds a message sequence
graph for each sliding window of CAN frames, annotates each node with the mean and
standard deviation of that ID's signals within the window, and classifies the graph
with a two-layer graph convolutional network.

## What you can run

| Task | Needs |
|---|---|
| Run the unit tests | Python packages only |
| Rebuild the tables and figures from the included results | Python packages only |
| Compute prevalence | The ROAD dataset |
| Create the capture files | The ROAD dataset and **ACTT** |
| Re-run the experiments | The ROAD dataset, **ACTT**, and the capture files |

**ACTT is not included in this repository.** See [ACTT](#actt) below.

## Reproducing the paper

1. [Set up the environment](#environment).
2. [Get the ROAD dataset](#data) and [ACTT](#actt).
3. [Create the capture files](#create-the-capture-files).
4. [Run the experiments](#run-the-experiments) (about 4.5 hours on a CPU).
5. [Rebuild the tables and figures](#rebuild-the-tables-and-figures).

Steps 1 and 5 alone rebuild every table and figure from the results in `results/`.
Run every command from the repository root.

## Environment

The paper's results were produced with Python 3.13.2 on macOS, on an Apple M4 CPU.
All training and timing runs on the CPU, so no GPU is needed.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

[requirements.txt](requirements.txt) pins the versions used for the paper:

| Package | Version |
|---|---|
| torch | 2.7.0 |
| torch-geometric | 2.6.1 |
| networkx | 3.4.2 |
| numpy | 2.2.5 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| cantools | 40.2.2 |
| matplotlib | 3.10.3 |

To create captures or re-run experiments, also install the two packages ACTT needs
that are not listed above:

```bash
pip install bitstring seaborn
```

Install these by name rather than from ACTT's own `requirements.txt`, which lists the
retired `sklearn` package name. It also lists `python-vpic`, which is not needed here
(see [Create the capture files](#create-the-capture-files)).

Check the installation:

```bash
python -m tests.test_logic
```

## Data

Download the [ROAD dataset](https://doi.org/10.1371/journal.pone.0296879) and keep its
original layout:

```text
road/
├── ambient/ambient_dyno_drive_winter.log
├── attacks/
│   ├── capture_metadata.json
│   └── <attack>.log
└── signal_extractions/DBC/anonymized.dbc
```

The experiments use all 13 ROAD masquerade captures and, as benign traffic, the
`ambient_dyno_drive_winter` capture. [configs/grid.csv](configs/grid.csv) lists them.
Injection intervals are read from `capture_metadata.json`, and signals are decoded
with `anonymized.dbc`.

## ACTT

PhantoMasq decodes CAN signals with ACTT (Automotive CAN Tokenization and
Translation), a CAN-analysis toolkit developed at Oak Ridge National Laboratory
([Verma et al., 2018](https://ieeexplore.ieee.org/abstract/document/8947694)). ACTT is not publicly released
and has no license, so it is not included here. To request access, contact the
authors.

PhantoMasq uses only two parts of ACTT:

- **`init_cancapture_from_canlog.py`** parses a ROAD `.log` file into a pickled
  `Capture` object (`capture.pkl`), which holds each CAN ID's frames and timestamps.
- **`CAN_objects.capture.MappedCapture.init_from_dbc`** decodes a `Capture` with the
  ROAD DBC into per-signal time series for every ID. PhantoMasq loads these in
  [phantomasq/data.py](phantomasq/data.py) (`load_mapped_capture`).

ACTT's reverse-engineering features are not used: every signal definition comes from
the ROAD DBC. The paper used ACTT commit `0cc31c0` (November 2020). Pass ACTT's `src/`
directory as `--actt_src`. It does not need to be installed.

## Create the capture files

Create one `capture.pkl` for each of the 14 captures in the grid:

```bash
python -m scripts.make_captures --road_dir /path/to/road --actt_src /path/to/actt/src --captures_dir /path/to/captures
```

This writes `/path/to/captures/<capture name>/capture.pkl`, using the folder names in
`configs/grid.csv`. It skips captures that already exist. It runs ACTT's
`init_cancapture_from_canlog.py` on each log, with three adjustments so that the
output matches the captures used in the paper:

- **No frames are dropped.** ACTT always skips a log's first and last line, which it
  expects to be a header and a footer. ROAD logs have neither, so the script adds a
  placeholder line at each end of a temporary copy of the log.
- **Folder names match the grid.** ACTT names each folder `<name>_<YYMMDD>_<HHMMSS>`
  after the log's start time in the local time zone. The paper's captures were
  created in US Eastern time, so the script runs ACTT with `TZ=America/New_York`.
- **No web lookup.** ACTT looks up each capture name as a vehicle VIN with the NHTSA
  vPIC web service. ROAD capture names are not VINs, so the script replaces the
  lookup with one that reports no match. `python-vpic` is therefore not needed.

We checked the result against the captures used in the paper. All 14 decode to
identical signal values and timestamps.

## Preprocessing

[phantomasq/data.py](phantomasq/data.py) (`make_can_df`) applies these steps to every
log, in this order:

1. **Parsing.** Each line of a ROAD log gives a timestamp, a CAN ID, and an 8-byte
   payload. As in ACTT, the parser skips each log's first and last line, so the first
   and last frames of each ROAD log are not used.
2. **Time.** Timestamps are shifted so each capture starts at 0, to match the
   injection intervals in `capture_metadata.json`.
3. **CAN ID filtering.** Frames with IDs above `0x700` are removed. In the 14 logs used
   here, the only such ID is `0xFFF`, which makes up 6.3% of frames. It is not defined
   in the ROAD DBC, so it has no signals to decode. The remaining 105 IDs are all
   defined in the DBC, so every graph node has signal features. The `0x700` cutoff
   follows the preprocessing of
   [Marfo et al.](https://github.com/billmj/GraphML-CONTROLLER-AREA-NETWORK-Attack-Detection).

After filtering, each window is processed as follows:

- **Graph construction.** Each window becomes a directed message sequence graph. Its
  nodes are the CAN IDs in the window, and each edge's weight counts how many times one
  ID was directly followed by another.
- **Node features.** Each node gets the mean and standard deviation of each of its
  decoded signals over the window's time span. Nodes with fewer signals than the
  largest ID are zero-padded, so every node has the same feature length.

Constant-valued IDs are kept. Their signals have a standard deviation of 0.

## Run the experiments

Re-run all 468 experiments (13 captures × 36 window configurations × 5 seeds):

```bash
python -m scripts.run_grid --road_dir /path/to/road --captures_dir /path/to/captures --actt_src /path/to/actt/src
```

For each grid row, this runs:

- `scripts.train_eval`, which trains and tests the model for each seed;
- `scripts.measure_ttw`, which measures end-to-end testing time per window.

The full sweep took about 4.5 hours on an Apple M4 CPU. Results go to
`results/rerun/<attack>.csv`, so the included results are not overwritten. Logs, JSON
outputs, and model checkpoints for each run go to `runs/`. If a sweep stops, run the
same command again and it continues where it stopped. Add `--dry_run` to print the
commands without running them.

To run one configuration:

```bash
python -m scripts.train_eval --road_dir /path/to/road --captures_dir /path/to/captures --actt_src /path/to/actt/src \
    --attack max_speedometer_attack_1_masquerade \
    --attack_capture road_attack_max_speedometer_attack_1_masquerade_060215_054000 \
    --ambient ambient_dyno_drive_winter \
    --ambient_capture road_ambient_dyno_drive_winter_030410_144000 \
    --window_size 150 --window_offset 50
```

Then run `python -m scripts.measure_ttw` with the same paths, capture, and window
options. Add `--help` to either script for all options. The defaults are the
hyperparameters used in the paper: 30 epochs, a learning rate of 0.01, 64 hidden
channels, a batch size of 64, and seeds 1–5.

## Rebuild the tables and figures

| Paper | Command | Output |
|---|---|---|
| Table II (end-to-end TTW) | `python -m scripts.summarize_results` | `results/table_ttw_summary.csv` |
| Table III (best configurations) | `python -m scripts.summarize_results` | `results/table_best_configurations.csv` |
| Table IV (comparison with Marfo et al.) | `python -m scripts.summarize_results --baseline_csv <file>` | `results/table_baseline_comparison.csv` |
| Fig. 4 (AUC-ROC and AUC-PR heatmaps) | `python -m scripts.make_figures` | `results/figures/auc_roc.png`, `results/figures/auc_pr.png` |
| Prevalence | `python -m scripts.compute_prevalence --road_dir /path/to/road` | `results/prevalence.csv` |

`summarize_results` also writes `results/all_results.csv`, which `make_figures` reads,
so run it first.

These commands read the included per-capture results in `results/captures/`. To build
the tables and figures from your own sweep instead, write them to a separate folder.
Do not write them into `results/rerun/` itself, because `summarize_results` reads every
CSV file in its `--results_dir`.

```bash
mkdir -p results/rerun/tables
```

```bash
python -m scripts.summarize_results --results_dir results/rerun --output_dir results/rerun/tables
```

```bash
python -m scripts.make_figures --results results/rerun/tables/all_results.csv --output_dir results/rerun/tables/figures
```

Table IV also needs `final_consolidated_performance_metrics.csv` from the
[Marfo et al. repository](https://github.com/billmj/GraphML-CONTROLLER-AREA-NETWORK-Attack-Detection).
Pass its path with `--baseline_csv`.

## Evaluation protocol

- **Labels:** each attack capture is split at its ROAD injection interval into
  benign-before, attack, and benign-after regions. Only windows in the attack region
  are malicious. The split is used to label evaluation data; a deployed detector would
  window the live stream directly.
- **Train/test split:** each region is split in time, 70% train and 30% test, before
  windowing, so no window crosses a train/test boundary.
- **Windows:** training and test windows contain exactly ω frames. End-to-end timing
  slides over the whole capture as a stream, so it also includes the capture's final,
  shorter windows.
- **Model selection:** there is no validation set. Hyperparameters were fixed in
  advance, and each model is scored once on the test set after its final epoch.
- **Seeds:** results are means over seeds 1–5.

## Reproducibility

Training on CPU is deterministic: on the same machine and package versions, AUC values
repeat exactly. Seeded MPS runs did not repeat, so CPU is the default. Other CPUs or
package versions may change AUC values slightly. Timing results vary from run to run
and across hardware.

## Repository layout

```text
phantomasq/   Parsing, labeling, windowing, graph construction, and the GCN
scripts/      Capture creation, experiments, timing, prevalence, tables, and figures
configs/      The 468-experiment grid
results/      Per-capture results, paper tables, prevalence, and figures
tests/        Tests for the splitting, windowing, and labeling logic
```
