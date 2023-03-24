import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# Linkpred con GNN
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.g_conv1 = GCNConv(in_channels * 2, hidden_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.g_conv2 = GCNConv(hidden_channels, out_channels)
        self.ln = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(out_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, edge_index, cutoff):
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        x = self.g_conv1(edge_features, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.g_conv2(x, edge_index)
        x = self.linear(x)
        x = self.sigmoid(x)

        rounded_values = torch.where(x < cutoff, torch.zeros_like(x), torch.ones_like(x))

        return x.squeeze(), rounded_values.squeeze().detach()

# Linkpred con MLP
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cutoff):
        out = self.linear1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        rounded_values = torch.where(out < cutoff, torch.zeros_like(out), torch.ones_like(out))

        return out.squeeze(), rounded_values.squeeze()
