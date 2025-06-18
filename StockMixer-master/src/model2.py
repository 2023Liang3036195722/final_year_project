import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self, stocks, time_steps, features):
        super(StockLSTM, self).__init__()
        self.stocks = stocks
        self.time_steps = time_steps
        self.features = features
        self.input_ln = nn.LayerNorm(features)
        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.bn = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, 1)

    def forward(self, inputs):
        # inputs的形状为 (B, N, T, F) 将其转换为 (B*N, T, F) 以适应LSTM的输入
        batch_size = inputs.size(0)
        inputs = inputs.view(-1, self.time_steps, self.features)
        normalized_inputs = self.input_ln(inputs)
        lstm_out, _ = self.lstm(normalized_inputs)
        last_output = lstm_out[:, -1, :]  # 形状: (B*N, 64)
        normalized_output = self.bn(last_output)
        output = self.fc(normalized_output)
        output = output.view(batch_size, self.stocks)
        return output