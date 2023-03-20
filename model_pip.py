import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class EdgeWeightPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(EdgeWeightPredictor, self).__init__()
        self.g_conv1 = GCNConv(in_channels * 2, in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.g_conv2 = GCNConv(in_channels, in_channels)
        self.ln2 = nn.LayerNorm(in_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_scores = self.g_conv1(edge_features, edge_index)
        edge_scores = self.bn1(edge_scores)
        edge_scores = self.relu(edge_scores)
        edge_scores = self.g_conv2(edge_scores, edge_index)
        edge_scores = self.ln2(edge_scores)
        edge_scores = self.relu(edge_scores)
        edge_scores = self.linear(edge_scores)
        # edge_scores = edge_scores - edge_scores.min()
        
        return edge_scores.squeeze()


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.g_conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.g_conv2 = GCNConv(hidden_channels, out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(out_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.g_conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.g_conv2(x, edge_index)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.linear(x)

        return x

    def decode(self, z, edge_label_index, cutoff):
        edge_values = (z[edge_label_index[0].to(torch.long)] * z[edge_label_index[1].to(torch.long)]).sum(dim=-1)
        rounded_values = torch.where(edge_values < cutoff, torch.zeros_like(edge_values), torch.ones_like(edge_values))

        return edge_values.squeeze(), rounded_values.squeeze().detach()

# Linkpred con MLP
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        # self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.ln2 = nn.LayerNorm(out_channels)

    def encode(self, x, edge_index):
        x = self.linear1(x)
        # x = self.bn1(x) Probar que tal funcionan
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.linear2(x)
        # x = self.ln2(x) Probar que tal funcionan
        x = self.relu(x)

        return x

    def decode(self, z, edge_label_index, cutoff):
        edge_values = (z[edge_label_index[0].to(torch.long)] * z[edge_label_index[1].to(torch.long)]).sum(dim=-1)
        rounded_values = torch.where(edge_values < cutoff, torch.zeros_like(edge_values), torch.ones_like(edge_values))

        return edge_values.squeeze(), rounded_values.squeeze().detach()

# Linkpred con LSTM
class LSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.g_lstm = torch.nn.LSTM(in_channels, hidden_channels)
        # self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.g_conv = GCNConv(hidden_channels, out_channels)
        # x = self.ln2(x) Probar que tal funcionan
        self.relu = nn.ReLU()

    def encode(self, x, edge_index):
        x = x.unsqueeze(0) # Agrega una dimensión temporal para la entrada de la capa LSTM
        x, _ = self.g_lstm(x)
        # x = self.bn1(x) Probar que tal funcionan
        x = x.squeeze(0) # Elimina la dimensión temporal adicional
        x = self.g_conv(x, edge_index)
        # x = self.ln2(x) Probar que tal funcionan
        x = self.relu(x)

        return x

    def decode(self, z, edge_label_index, cutoff):
        edge_values = (z[edge_label_index[0].to(torch.long)] * z[edge_label_index[1].to(torch.long)]).sum(dim=-1)
        rounded_values = torch.where(edge_values < cutoff, torch.zeros_like(edge_values), torch.ones_like(edge_values))

        return edge_values.squeeze(), rounded_values.squeeze().detach()