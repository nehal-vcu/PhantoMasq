"""Train and evaluate PhantoMasq on one attack capture and one (omega, delta) setting.

Hyperparameters are fixed in advance: there is no validation set, the model is
trained for the full epoch budget, and the test set is scored once on the
final-epoch model, repeated over several seeds.
"""

import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from phantomasq.data import (
    build_msg,
    load_injection_interval,
    load_mapped_capture,
    make_can_df,
    partition_attack_regions,
    road_paths,
    slice_region_windows,
    temporal_split,
)
from phantomasq.model import (
    GCN,
    annotate_window,
    graph_to_data,
    max_signals_for_nodes,
    pick_device,
    synchronize,
)


def region_examples(region_df, mapped_capture, label, window_size, window_offset, max_signals):
    examples = []
    for window, t_start, t_end in slice_region_windows(region_df, window_size, window_offset):
        graph = build_msg(window)
        annotated = annotate_window(graph, mapped_capture, t_start, t_end, max_signals)
        examples.append(graph_to_data(graph, annotated, label))
    return examples


def build_datasets(args):
    paths = road_paths(args.road_dir, args.attack, args.ambient)
    inj_start, inj_end = load_injection_interval(paths["metadata"], args.attack)

    ambient_capture = load_mapped_capture(args.actt_src, args.captures_dir,
                                          args.ambient_capture, paths["dbc"])
    attack_capture = load_mapped_capture(args.actt_src, args.captures_dir,
                                         args.attack_capture, paths["dbc"])
    ambient_df = make_can_df(paths["ambient_log"])
    attack_df = make_can_df(paths["attack_log"])

    max_signals = max(
        max_signals_for_nodes(ambient_capture, ambient_df["pid"].unique().tolist()),
        max_signals_for_nodes(attack_capture, attack_df["pid"].unique().tolist()),
    )

    before, attack, after = partition_attack_regions(attack_df, inj_start, inj_end)
    print(f"[regions] before={len(before)} attack={len(attack)} after={len(after)} frames")
    if len(attack) == 0:
        raise SystemExit("The attack region is empty: check the injection interval.")

    regions = [(ambient_df, ambient_capture, 0), (before, attack_capture, 0),
               (attack, attack_capture, 1), (after, attack_capture, 0)]
    train, test = [], []
    for region_df, capture, label in regions:
        region_train, region_test = temporal_split(region_df)
        train += region_examples(region_train, capture, label,
                                 args.window_size, args.window_offset, max_signals)
        test += region_examples(region_test, capture, label,
                                args.window_size, args.window_offset, max_signals)

    balance = {"train": class_balance("train", train), "test": class_balance("test", test)}
    return train, test, max_signals, balance


def class_balance(name, examples):
    positives = sum(int(example.y.item()) for example in examples)
    ratio = positives / len(examples) if examples else 0.0
    print(f"[{name}] graphs={len(examples)} positives={positives} prevalence={ratio:.3f}")
    return {"graphs": len(examples), "positives": positives, "attack_ratio": ratio}


def evaluate(model, loader, device):
    """Return (AUC-ROC, AUC-PR); AUC-PR is computed as average precision."""
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def train_one_seed(seed, train, test, in_channels, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GCN(in_channels, hidden_channels=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
            criterion(logits, batch.y).backward()
            optimizer.step()

    test_roc, test_pr = evaluate(model, test_loader, device)
    return model, test_roc, test_pr


def measure_model_ttw(model, test, device):
    """Measure device transfer plus forward pass for one graph at a time.

    A PyG DataLoader must not be used here: its per-item collation would be
    timed as part of inference. The warm-up keeps one-time initialization out
    of the first window's timing.
    """
    model.eval()
    with torch.no_grad():
        for example in test[:5]:
            example = example.clone()
            example.batch = torch.zeros(example.num_nodes, dtype=torch.long)
            example = example.to(device)
            model(example.x, example.edge_index, example.batch, example.edge_weight)
    synchronize(device)

    total = 0.0
    with torch.no_grad():
        for example in test:
            example = example.clone()
            example.batch = torch.zeros(example.num_nodes, dtype=torch.long)
            synchronize(device)
            start = time.perf_counter()
            example = example.to(device)
            model(example.x, example.edge_index, example.batch, example.edge_weight)
            synchronize(device)
            total += time.perf_counter() - start

    return total / len(test), len(test) / total


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road_dir", required=True, help="Root of the ROAD dataset")
    parser.add_argument("--captures_dir", required=True, help="Directory of ACTT capture folders")
    parser.add_argument("--actt_src", required=True, help="ACTT src directory")
    parser.add_argument("--attack", required=True, help="Attack log name, without .log")
    parser.add_argument("--attack_capture", required=True, help="Attack capture folder name")
    parser.add_argument("--ambient", required=True, help="Ambient log name, without .log")
    parser.add_argument("--ambient_capture", required=True, help="Ambient capture folder name")
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--window_offset", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--output_json", default="train_eval.json")
    parser.add_argument("--save_model", default="model.pt",
                        help="Checkpoint of the last seed, used by scripts.measure_ttw")
    # Seeded CPU runs repeat exactly. Seeded MPS runs did not.
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--ttw_device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    ttw_device = pick_device(args.ttw_device)

    train, test, max_signals, balance = build_datasets(args)
    in_channels = train[0].x.shape[1]
    seeds = [int(seed) for seed in args.seeds.split(",")]

    rocs, prs = [], []
    for seed in seeds:
        model, test_roc, test_pr = train_one_seed(seed, train, test, in_channels, args, device)
        rocs.append(float(test_roc))
        prs.append(float(test_pr))
        print(f"[seed {seed}] AUC-ROC={test_roc:.4f} AUC-PR={test_pr:.4f}")

    ttw, throughput = measure_model_ttw(model.to(ttw_device), test, ttw_device)

    torch.save({
        "state_dict": model.state_dict(),
        "in_channels": in_channels,
        "hidden_channels": args.hidden,
        "max_signals": max_signals,
    }, args.save_model)

    results = {
        "attack": args.attack,
        "window_size": args.window_size,
        "window_offset": args.window_offset,
        "epochs": args.epochs,
        "seeds": seeds,
        "splits": balance,
        "test_auc_mean": float(np.mean(rocs)),
        "test_auc_std": float(np.std(rocs)),
        "test_auc_per_seed": rocs,
        "test_auc_pr_mean": float(np.mean(prs)),
        "test_auc_pr_std": float(np.std(prs)),
        "test_auc_pr_per_seed": prs,
        "model_side_ttw_ms": ttw * 1000,
        "throughput_windows_per_sec": throughput,
        "num_test_windows": len(test),
        "ttw_device": str(ttw_device),
    }
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"AUC-ROC {results['test_auc_mean']:.4f} +/- {results['test_auc_std']:.4f}")
    print(f"AUC-PR  {results['test_auc_pr_mean']:.4f} +/- {results['test_auc_pr_std']:.4f}")
    print(f"Model-side TTW {ttw * 1000:.3f} ms/window on {ttw_device}")


if __name__ == "__main__":
    main()
