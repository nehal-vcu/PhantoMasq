"""Graph annotation, PyG conversion and the GCN model."""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


def max_signals_for_nodes(mapped_capture, node_ids):
    """Largest signal count among the given IDs in one capture."""
    return max((len(mapped_capture.mapped_payload_dict[nid].signal_list) for nid in node_ids),
               default=0)


def annotate_window(graph, mapped_capture, t_start, t_end, max_signals):
    """Compute per-node [mean, std] for each signal within the window's time span.

    Used for benign and malicious windows alike so both classes share one
    feature definition. Nodes are padded with [0, 0] to max_signals signals so
    every graph has the same feature width.
    """
    annotated = {}
    for node in graph.nodes():
        signals = mapped_capture.mapped_payload_dict[node].signal_list
        annotated[node] = {}
        for j, signal in enumerate(signals):
            times = np.asarray(signal.times)
            values = np.asarray(signal.values)
            in_window = values[(times >= t_start) & (times <= t_end)]
            if in_window.size > 0:
                annotated[node][f"s{j}"] = [float(np.mean(in_window)), float(np.std(in_window))]
            else:
                annotated[node][f"s{j}"] = [0.0, 0.0]
        for j in range(len(signals), max_signals):
            annotated[node][f"s{j}"] = [0.0, 0.0]
    return annotated


def graph_to_data(graph, annotated, label):
    """Convert an annotated MSG into a PyG Data object."""
    node_ids = sorted(graph.nodes())
    index = {node: i for i, node in enumerate(node_ids)}

    edges = [(index[src], index[dst]) for src, dst in graph.edges()]
    weights = [float(attrs.get("weight", 1.0)) for _, _, attrs in graph.edges(data=True)]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)

    # s10 sorts before s2 here; the order is arbitrary but must stay fixed
    # because trained models depend on it.
    features = []
    for node in node_ids:
        row = []
        for key in sorted(annotated.get(node, {})):
            row.extend(annotated[node][key])
        features.append(row)

    return Data(x=torch.tensor(features, dtype=torch.float),
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=torch.tensor([label], dtype=torch.long))


class GCN(torch.nn.Module):
    """Two-layer GCN with mean pooling and a linear classifier."""

    def __init__(self, in_channels, hidden_channels=64, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        return self.lin(global_mean_pool(x, batch))


def pick_device(requested="cpu"):
    """Resolve a device name; 'auto' prefers cuda, then mps, then cpu."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    """Wait for queued accelerator work so timers measure completed work."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
