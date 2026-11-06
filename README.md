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
| Re-run the experiments | The ROAD dataset, **ACTT, and ACTT capture files** |

**ACTT and the `capture.pkl` capture files are not included in this repository.**
Training and end-to-end timing decode CAN signals through ACTT's `MappedCapture`, so
these steps cannot run without them.

## Setup

Tested with Python 3.13.2 on macOS. Run every command from the repository root.

```bash
pip install -r requirements.txt
```

To re-run the experiments, also install ACTT's own requirements.

## Data

Prevalence and the experiments read the [ROAD dataset](https://doi.org/10.1371/journal.pone.0296879)
in its original layout:

```text
road/
├── ambient/ambient_dyno_drive_winter.log
├── attacks/
│   ├── capture_metadata.json
│   └── <attack>.log
└── signal_extractions/DBC/anonymized.dbc
```

The experiments also need the capture folders named in [configs/grid.csv](configs/grid.csv),
each containing a `capture.pkl`, inside the directory passed as `--captures_dir`.

## Usage

Check the code:

```bash
python -m tests.test_logic
```

Rebuild the tables and figures from the included results:

```bash
python -m scripts.summarize_results
python -m scripts.make_figures
```

The baseline comparison table also needs `final_consolidated_performance_metrics.csv`
from the [Marfo et al. repository](https://github.com/billmj/GraphML-CONTROLLER-AREA-NETWORK-Attack-Detection):

```bash
python -m scripts.summarize_results --baseline_csv /path/to/final_consolidated_performance_metrics.csv
```

Compute prevalence:

```bash
python -m scripts.compute_prevalence --road_dir /path/to/road
```

Re-run all 468 experiments (13 captures × 36 window configurations × 5 seeds):

```bash
python -m scripts.run_grid --road_dir /path/to/road --captures_dir /path/to/captures --actt_src /path/to/actt/src
```

The full sweep took about 4.5 hours on an Apple M4 CPU. Results go to `results/rerun/`,
so the included results are not overwritten. If a sweep stops, run the same command
again and it continues from where it stopped.

To run one experiment, use `python -m scripts.train_eval`, then
`python -m scripts.measure_ttw`. Add `--help` to either for its options.

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
repeat exactly. Seeded MPS runs did not repeat, so CPU is the default. Timing results
vary from run to run and across hardware.

## Repository layout

```text
phantomasq/   Parsing, labeling, windowing, graph construction, and the GCN
scripts/      Experiments, timing, prevalence, tables, and figures
configs/      The 468-experiment grid
results/      Per-capture results, paper tables, prevalence, and figures
tests/        Tests for the splitting, windowing, and labeling logic
```
