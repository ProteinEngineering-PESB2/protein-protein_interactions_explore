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
    
# Linkpred con LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cutoff):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear1(out[:, -1, :])
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        rounded_values = torch.where(out < cutoff, torch.zeros_like(out), torch.ones_like(out))

        return out.squeeze(), rounded_values.squeeze()
    
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
    
# Edge Regressor con LSTM
class LSTMModelEdgeRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear1(out[:, -1, :])
        out = self.bn(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.ln(out)
        out = self.relu(out)
        out = self.linear3(out)

        return out.squeeze()

# #  CNN
# class CnnC(nn.Module):
#     """
#     mode = ("binary", "classification", "regression")

#     Utiliza Dobles Convoluciones 1D, intercaladas con capas max pooling.

#     Incorpora una capa Dropout con rate 0.25.

#     Finaliza con flatten y dense.
#     """
#     def __init__(self, input_shape, num_classes):
#         super().__init__()

#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2)
#         self.relu2 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=2)
#         self.dropout1 = nn.Dropout(0.25)

#         self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
#         self.relu3 = nn.ReLU()
#         self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
#         self.relu4 = nn.ReLU()
#         self.pool2 = nn.AvgPool1d(kernel_size=4)
#         self.dropout2 = nn.Dropout(0.25)
        
#         self.conv5 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2)
#         self.relu5 = nn.ReLU()
#         self.conv6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2)
#         self.relu6 = nn.ReLU()
#         self.pool3 = nn.MaxPool1d(kernel_size=4)
#         self.dropout3 = nn.Dropout(0.25)

#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(in_features=64, out_features=64)
#         self.tanh = nn.Tanh()
#         self.fc2 = nn.Linear(in_features=64, out_features=num_classes)
#         self.softmax = nn.Softmax(dim=1)
    
#     def forward(self, x):
#         x = x.unsqueeze(1)

#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool1(x)
#         x = self.dropout1(x)

#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.conv4(x)
#         x = self.relu4(x)
#         x = self.pool2(x)
#         x = self.dropout2(x)

#         x = self.conv5(x)
#         x = self.relu5(x)
#         x = self.conv6(x)
#         x = self.relu6(x)
#         x = self.pool3(x)
#         x = self.dropout3(x)

#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.tanh(x)
#         x = self.fc2(x)
#         x = self.softmax(x)

#         return x.squeeze()
