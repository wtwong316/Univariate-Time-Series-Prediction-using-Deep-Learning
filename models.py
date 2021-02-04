import torch
import torch.nn as nn

import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DNN(nn.Module):
    """Deep Neural Network"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x.squeeze(dim=2)
        out = self.main(x)
        return out


class RNN(nn.Module):
    """Vanilla RNN"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


class LSTM(nn.Module):
    """Long Short Term Memory"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        if self.bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


class GRU(nn.Module):
    """Gat e Recurrent Unit"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


class RecursiveLSTM(nn.Module):
    """Recursive LSTM"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RecursiveLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        pred = torch.empty([x.shape[0], self.output_size]).to(device)
        for i in range(self.output_size):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)

            pred[:, i] = torch.squeeze(out, -1)
            out = torch.unsqueeze(out, -1)
            x = torch.cat([x, out], 1)[:, 1:, :]

        return pred


class AttentionLSTM(nn.Module):
    """LSTM with Attention"""
    def __init__(self, input_size, key, query, value, hidden_size, num_layers, output_size, bidirectional=False):
        super(AttentionLSTM, self).__init__()

        self.input_size = input_size
        self.key = key
        self.query = query
        self.value = value
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.query = nn.Linear(input_size, query)
        self.key = nn.Linear(input_size, key)
        self.value = nn.Linear(input_size, value)

        self.attn = nn.Linear(value, input_size)
        self.scale = math.sqrt(query)

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        dot_product = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale
        scores = torch.softmax(dot_product, dim=-1)
        scaled_x = torch.matmul(scores, V) + x

        new_x = self.attn(scaled_x) + x
        out, _ = self.lstm(new_x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


class CNN(nn.Module):
    """Convolutional Neural Networks"""
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(out_channels, 10),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        out = self.main(x)
        return out