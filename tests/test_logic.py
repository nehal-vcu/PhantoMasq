"""Tests for the windowing, splitting and labeling logic.

Run from the repository root with: python -m tests.test_logic
No dataset, ACTT installation or GPU is needed.
"""

import numpy as np
import pandas as pd

from phantomasq.data import (
    build_msg,
    partition_attack_regions,
    slice_region_windows,
    temporal_split,
)


def synthetic_capture(n=1000, hz=100.0, pids=(0xD0, 0x6E0, 0x4E7, 0x100)):
    return pd.DataFrame({
        "sample_number": range(n),
        "pid": [pids[i % len(pids)] for i in range(n)],
        "time": np.arange(n) / hz,
        "data": ["0" * 16] * n,
    })


def sample_ids(region, omega, delta):
    ids = set()
    for window, _, _ in slice_region_windows(region, omega, delta):
        ids.update(window.sample_number.tolist())
    return ids


def test_partition_matches_injection_interval():
    capture = synthetic_capture()
    before, attack, after = partition_attack_regions(capture, 3.0, 6.0)
    assert before.time.max() < 3.0
    assert attack.time.min() >= 3.0 and attack.time.max() <= 6.0
    assert after.time.min() > 6.0
    assert len(before) + len(attack) + len(after) == len(capture)


def test_temporal_split_is_contiguous_and_disjoint():
    train, test = temporal_split(synthetic_capture(n=200), 0.7)
    assert len(train) == 140 and len(test) == 60
    assert train.time.max() < test.time.min()
    assert not set(train.sample_number) & set(test.sample_number)


def test_overlapping_windows_never_share_frames_across_splits():
    train, test = temporal_split(synthetic_capture())
    assert not sample_ids(train, 100, 50) & sample_ids(test, 100, 50)


def test_windows_hold_exactly_omega_frames():
    region = synthetic_capture(n=437)
    windows = slice_region_windows(region, 100, 50)
    assert len(windows) == (437 - 100) // 50 + 1
    assert all(len(window) == 100 for window, _, _ in windows)


def test_keep_partial_includes_short_tail_windows():
    region = synthetic_capture(n=437)
    windows = slice_region_windows(region, 100, 50, keep_partial=True)
    assert len(windows) == int(np.ceil(437 / 50))
    assert len(windows[-1][0]) < 100


def test_windows_stay_inside_their_region():
    region = synthetic_capture(n=300).iloc[100:200].reset_index(drop=True)
    for _, t_start, t_end in slice_region_windows(region, 30, 15):
        assert t_start >= region.time.min() and t_end <= region.time.max()


def test_positive_windows_lie_inside_the_injection_interval():
    before, attack, after = partition_attack_regions(synthetic_capture(), 3.0, 6.0)
    positives = slice_region_windows(attack, 100, 50)
    assert positives
    assert all(t_start >= 3.0 and t_end <= 6.0 for _, t_start, t_end in positives)
    assert slice_region_windows(before, 100, 50) and slice_region_windows(after, 100, 50)


def test_msg_edges_count_transitions():
    graph = build_msg(synthetic_capture(n=5, pids=(1, 2)))
    assert set(graph.nodes()) == {1, 2}
    assert graph[1][2]["weight"] == 2 and graph[2][1]["weight"] == 2


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")
