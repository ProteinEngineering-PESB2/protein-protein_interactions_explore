import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# Linkpred con GNN
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.g_conv1 = GCNConv(in_channels, hidden_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.g_conv2 = GCNConv(hidden_channels, out_channels)
        self.ln = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(out_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.g_conv1(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.g_conv2(x, edge_index)
        x = self.ln(x)
        x = self.relu(x)
        x = self.linear(x)

        return x

    def decode(self, z, edge_label_index, cutoff):
        edge_values = (z[edge_label_index[0].to(torch.long)] * z[edge_label_index[1].to(torch.long)]).sum(dim=-1)
        rounded_values = torch.where(edge_values < cutoff, torch.zeros_like(edge_values), torch.ones_like(edge_values))

        return edge_values.squeeze(), rounded_values.squeeze().detach()

# Linkpred con MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
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
