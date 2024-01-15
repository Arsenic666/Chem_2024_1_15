# Modules for Multi-modal Reaction Yield Prediction Model (MMYield)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import numpy as np

# graph module

# GCN_module
class GCN_module(nn.Module):
    def __init__(self, node_feature_num, channels):
        super(GCN_module, self).__init__()

        self.conv1 = pyg_nn.GCNConv(node_feature_num, channels[0])
        self.norm1 = nn.BatchNorm1d(channels[0])
        self.conv2 = pyg_nn.GCNConv(channels[0], channels[1])

    def forward(self, data):
        x = data["x"].clone().detach().float()
        edge_index = data["edge_index"].clone().detach()
        batch = data["batch"].clone().detach()

        # GCN
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)

        x = torch.cat([x1, x2], 1)
        # Pooling
        x_mean = pyg_nn.global_mean_pool(x, batch=batch)
        x_max = pyg_nn.global_max_pool(x, batch=batch)
        x = torch.cat([x_mean, x_max], 1)
        return x

# GAT_module
class GAT_module(nn.Module):
    def __init__(self, node_feature_num, channels, heads):
        super(GAT_module, self).__init__()
        self.conv1 = pyg_nn.GATConv(node_feature_num, channels[0], heads=heads)
        self.norm1 = nn.BatchNorm1d(channels[0] * heads)
        self.conv2 = pyg_nn.GATConv(channels[0] * heads, channels[1], heads=heads)

    def forward(self, data):
        x = data["x"].clone().detach().float()
        edge_index = data["edge_index"].clone().detach()
        batch = data["batch"].clone().detach()

        # GAT
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)

        x = torch.cat([x1, x2], 1)
        # Pooling
        x_mean = pyg_nn.global_mean_pool(x, batch=batch)
        x_max = pyg_nn.global_max_pool(x, batch=batch)
        x = torch.cat([x_mean, x_max], 1)

        return x

# SAGE_module
class SAGE_module(nn.Module):
    def __init__(self, node_feature_num, channels):
        super(SAGE_module, self).__init__()
        self.conv1 = pyg_nn.SAGEConv(node_feature_num, channels[0])
        self.norm1 = nn.BatchNorm1d(channels[0])

        self.conv2 = pyg_nn.SAGEConv(channels[0], channels[1])
        self.norm2 = nn.BatchNorm1d(channels[1])

        self.conv3 = pyg_nn.SAGEConv(channels[1], channels[2])
        self.norm3 = nn.BatchNorm1d(channels[2])

    def forward(self, data):
        x = data["x"].clone().detach().float()
        edge_index = data["edge_index"].clone().detach()
        batch = data["batch"].clone().detach()

        # SAGE
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index)
        x3 = self.norm3(x3)
        x3 = F.relu(x3)

        x = torch.cat([x1, x2, x3], 1)
        # Pooling
        x = pyg_nn.global_mean_pool(x, batch=batch)
        return x


# nlp module
class smi2vec_module(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(smi2vec_module, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入时seq_len与batch_size顺序颠倒
            bidirectional=True
        )
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0) # x is input, size (seq_len, batch, input_size)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def module_selector(type, params):
    # graph module
    if type == "GCN_module":
        # params in NN
        for key in params:
            if "node_feature_num" in key:
                node_feature_num = params[key]
            if "channels" in key:
                channels = params[key]
        output_size = 0
        for i in channels:
            output_size += i
        return GCN_module(node_feature_num, channels), output_size * 2 # module & shape of output vector

    if type == "GAT_module":
        for key in params:
            # params in NN
            if "node_feature_num" in key:
                node_feature_num = params[key]
            if "channels" in key:
                channels = params[key]
            if "heads" in key:
                heads = params[key]
        # output vec shape
        output_size = 0
        for i in channels:
            output_size += i * heads
        return GAT_module(node_feature_num, channels, heads), output_size * 2 # module & shape of output vector

    if type == "SAGE_module":
        # params in NN
        for key in params:
            if "node_feature_num" in key:
                node_feature_num = params[key]
            if "channels" in key:
                channels = params[key]
        # output vec shape
        output_size = 0
        for i in channels:
            output_size += i
        return SAGE_module(node_feature_num, channels), output_size # module & shape of output vector

    # nlp module
    if type == "smi2vec_module":
        # params in NN
        for key in params:
            if "input_size" in key:
                input_size = params[key]
            if "hidden_size" in key:
                hidden_size = params[key]
            if "num_layers" in key:
                num_layers = params[key]
            if "output_size" in key:
                output_size = params[key]
        return smi2vec_module(input_size, hidden_size, num_layers, output_size), output_size # module & shape of output vector


# Model Evaluation Function
def RMSE(pred,true):
    diff_2 = (pred - true)**2
    return np.sqrt(diff_2.mean())

def R2(pred, true):
    u = ((true - pred) ** 2).sum()
    v = ((true - true.mean()) ** 2).sum()
    r2 = 1 - u / v
    return r2