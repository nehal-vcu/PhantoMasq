"""Data handling for PhantoMasq: log parsing, labeling, splitting and windowing.

Evaluation data is built per region. Each attack capture is partitioned at its
injection interval into benign-before, attack-interval and benign-after regions;
the ambient capture is one benign region. Every region is split temporally into
train and test BEFORE windowing, so no window can cross a train/test boundary
or an injection boundary.
"""

import json
import os
import sys

import networkx as nx
import pandas as pd


def road_paths(road_dir, attack, ambient):
    """Locate one experiment's inputs inside the public ROAD dataset layout."""
    return {
        "attack_log": os.path.join(road_dir, "attacks", f"{attack}.log"),
        "ambient_log": os.path.join(road_dir, "ambient", f"{ambient}.log"),
        "metadata": os.path.join(road_dir, "attacks", "capture_metadata.json"),
        "dbc": os.path.join(road_dir, "signal_extractions", "DBC", "anonymized.dbc"),
    }


def load_injection_interval(metadata_path, attack):
    """Return the ground-truth (start, end) injection interval, in seconds."""
    with open(metadata_path) as f:
        start, end = json.load(f)[attack]["injection_interval"]
    return float(start), float(end)


def load_mapped_capture(actt_src, captures_dir, capture_name, dbc_path):
    """Decode one capture.pkl with the DBC using ACTT, which is not bundled here."""
    if actt_src not in sys.path:
        sys.path.append(actt_src)
    import CAN_objects  # noqa: F401  (needed to unpickle captures)
    from CAN_objects.capture import MappedCapture
    from generalFunctions import unpickle

    capture = unpickle(os.path.join(captures_dir, capture_name, "capture.pkl"))
    return MappedCapture.init_from_dbc(capture, dbc_path)


def make_can_df(log_path):
    """Parse a ROAD .log file into [sample_number, pid, time, data].

    Time is normalized so the capture starts at 0, matching the metadata's
    injection intervals. Frames with IDs above 0x700 are excluded.
    """
    can_df = pd.read_fwf(
        log_path,
        delimiter=" #()",
        skiprows=1,
        skipfooter=1,
        usecols=[0, 2, 3],
        dtype={0: "float64", 1: str, 2: str},
        names=["time", "pid", "data"],
    )
    can_df.pid = can_df.pid.apply(lambda x: int(x, 16))
    can_df.data = can_df.data.apply(lambda x: x.zfill(16))
    can_df.time = can_df.time - can_df.time.min()
    can_df = can_df[can_df.pid <= 0x700].reset_index(drop=True)
    can_df["sample_number"] = range(len(can_df))
    return can_df[["sample_number", "pid", "time", "data"]]


def partition_attack_regions(can_df, inj_start, inj_end):
    """Split an attack capture into (benign_before, attack, benign_after)."""
    before = can_df[can_df.time < inj_start]
    attack = can_df[(can_df.time >= inj_start) & (can_df.time <= inj_end)]
    after = can_df[can_df.time > inj_end]
    return (before.reset_index(drop=True),
            attack.reset_index(drop=True),
            after.reset_index(drop=True))


def temporal_split(region_df, train_frac=0.7):
    """Split a time-ordered region into contiguous (train, test) slices."""
    n_train = min(int(round(len(region_df) * train_frac)), len(region_df))
    return (region_df.iloc[:n_train].reset_index(drop=True),
            region_df.iloc[n_train:].reset_index(drop=True))


def slice_region_windows(region_df, window_size, window_offset, keep_partial=False):
    """Slide a window over one contiguous region.

    Returns a list of (window_df, t_start, t_end). Tail windows holding fewer
    than window_size frames are dropped unless keep_partial is set. They must
    be dropped when building datasets: short regions carry proportionally more
    of them, which would correlate graph size with the class label.
    """
    windows = []
    start = 0
    while start < len(region_df):
        window = region_df.iloc[start:start + window_size]
        if len(window) < window_size and not keep_partial:
            break
        windows.append((window, float(window["time"].min()), float(window["time"].max())))
        start += window_offset
    return windows


def build_msg(window_df):
    """Build a directed message sequence graph for one window.

    Nodes are the unique IDs in the window; edge weights count consecutive
    transitions between IDs.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(window_df["pid"].unique().tolist())
    pids = window_df["pid"].to_numpy()
    for src, dst in zip(pids[:-1], pids[1:]):
        if graph.has_edge(src, dst):
            graph[src][dst]["weight"] += 1
        else:
            graph.add_edge(src, dst, weight=1)
    return graph
