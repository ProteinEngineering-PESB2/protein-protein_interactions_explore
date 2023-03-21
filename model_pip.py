import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


# Edge regressor GNN
class EdgeWeightPredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
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

# Edge regressor
class EdgeWeightPredictorMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeWeightPredictorMLP, self).__init__()
        self.linear1 = nn.Linear(in_channels * 2, in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_channels, in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_scores = self.linear1(edge_features)
        edge_scores = self.bn1(edge_scores)
        edge_scores = self.relu1(edge_scores)
        edge_scores = self.linear2(edge_scores)
        edge_scores = self.bn2(edge_scores)
        edge_scores = self.relu2(edge_scores)
        edge_scores = self.linear3(edge_scores)
        
        return edge_scores.squeeze()

# Linkpred con GNN
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
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU()
        self.ln2 = nn.LayerNorm(out_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.ln2(x)

        return x

    def decode(self, z, edge_label_index, cutoff):
        edge_values = (z[edge_label_index[0].to(torch.long)] * z[edge_label_index[1].to(torch.long)]).sum(dim=-1)
        rounded_values = torch.where(edge_values < cutoff, torch.zeros_like(edge_values), torch.ones_like(edge_values))

        return edge_values.squeeze(), rounded_values.squeeze().detach()

# Linkpred LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.bn1(out[:, -1, :])
        out = self.linear(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    
    def decode(self, z, edge_label_index, cutoff):
        edge_values = (z[edge_label_index[0].to(torch.long)] * z[edge_label_index[1].to(torch.long)]).sum(dim=-1)
        rounded_values = torch.where(edge_values < cutoff, torch.zeros_like(edge_values), torch.ones_like(edge_values))

        return edge_values.squeeze(), rounded_values.squeeze().detach()
