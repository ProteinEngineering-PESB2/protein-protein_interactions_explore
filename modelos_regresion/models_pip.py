import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


# Edge regressor GNN
class EdgeWeightPredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeWeightPredictor, self).__init__()
        self.g_conv1 = GCNConv(in_channels * 2, in_channels)
        self.bn = nn.BatchNorm1d(in_channels)
        self.g_conv2 = GCNConv(in_channels, in_channels)
        self.ln = nn.LayerNorm(in_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_scores = self.g_conv1(edge_features, edge_index)
        edge_scores = self.bn(edge_scores)
        edge_scores = self.relu(edge_scores)
        edge_scores = self.g_conv2(edge_scores, edge_index)
        edge_scores = self.ln(edge_scores)
        edge_scores = self.relu(edge_scores)
        edge_scores = self.linear(edge_scores)
        # edge_scores = edge_scores - edge_scores.min()
        
        return edge_scores.squeeze()

# Edge regressor MLP
class MLPEdgeRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
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
