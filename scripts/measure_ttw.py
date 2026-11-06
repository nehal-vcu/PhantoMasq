"""Measure end-to-end Testing Time per Window (TTW) over a full attack capture.

Each timed window covers MSG construction, windowed annotation, PyG conversion,
device transfer and the forward pass. Decoding the capture is one-time setup
and is not timed. Windows slide over the capture as a continuous stream, as they
would at deployment, so the capture's final, shorter windows are included.
"""

import argparse
import json
import time

import torch

from phantomasq.data import (
    build_msg,
    load_mapped_capture,
    make_can_df,
    road_paths,
    slice_region_windows,
)
from phantomasq.model import GCN, annotate_window, graph_to_data, pick_device, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road_dir", required=True, help="Root of the ROAD dataset")
    parser.add_argument("--captures_dir", required=True, help="Directory of ACTT capture folders")
    parser.add_argument("--actt_src", required=True, help="ACTT src directory")
    parser.add_argument("--attack", required=True, help="Attack log name, without .log")
    parser.add_argument("--attack_capture", required=True, help="Attack capture folder name")
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--window_offset", type=int, required=True)
    parser.add_argument("--model_path", default="model.pt")
    parser.add_argument("--output_json", default="measure_ttw.json")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = pick_device(args.device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = GCN(checkpoint["in_channels"], checkpoint["hidden_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    paths = road_paths(args.road_dir, args.attack, "")
    capture = load_mapped_capture(args.actt_src, args.captures_dir,
                                  args.attack_capture, paths["dbc"])
    can_df = make_can_df(paths["attack_log"])
    max_signals = checkpoint["max_signals"]

    total = 0.0
    windows = slice_region_windows(can_df, args.window_size, args.window_offset,
                                   keep_partial=True)
    for window, t_start, t_end in windows:
        synchronize(device)
        start = time.perf_counter()

        graph = build_msg(window)
        annotated = annotate_window(graph, capture, t_start, t_end, max_signals)
        data = graph_to_data(graph, annotated, label=1)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        data = data.to(device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.batch, data.edge_weight)
            torch.softmax(logits, dim=1)

        synchronize(device)
        total += time.perf_counter() - start

    ttw_ms = total / len(windows) * 1000
    throughput = len(windows) / total
    with open(args.output_json, "w") as f:
        json.dump({
            "attack": args.attack,
            "window_size": args.window_size,
            "window_offset": args.window_offset,
            "num_windows": len(windows),
            "ttw_ms": ttw_ms,
            "throughput_windows_per_sec": throughput,
            "device": str(device),
        }, f, indent=4)
    print(f"End-to-end TTW {ttw_ms:.3f} ms/window, {throughput:.1f} windows/s on {device}")


if __name__ == "__main__":
    main()
