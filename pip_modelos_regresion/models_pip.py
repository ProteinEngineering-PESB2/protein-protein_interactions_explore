import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


# Edge regressor GNN
class EdgeWeightPredictor(nn.Module):
    def __init__(self, in_channels,  output_size):
        super(EdgeWeightPredictor, self).__init__()
        self.g_conv1 = GCNConv(in_channels * 2, in_channels)
        self.bn = nn.BatchNorm1d(in_channels)
        self.g_conv2 = GCNConv(in_channels, output_size)
        self.linear = nn.Linear(output_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_scores = self.g_conv1(edge_features, edge_index)
        edge_scores = self.bn(edge_scores)
        edge_scores = self.relu(edge_scores)
        edge_scores = self.g_conv2(edge_scores, edge_index)
        edge_scores = self.linear(edge_scores)
        
        return edge_scores.squeeze()

# Edge regressor MLP
class MLPEdgeRegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_size):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.ln = nn.LayerNorm(hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.ln(out)
        out = self.relu(out)
        out = self.linear3(out)
        
        return out.squeeze()
