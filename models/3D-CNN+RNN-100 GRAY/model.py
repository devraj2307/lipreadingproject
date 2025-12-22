import torch
import torch.nn as nn
import torch.nn.functional as F

class LipNet(nn.Module):
    def __init__(self, num_classes=100, rnn_hidden_size=256, rnn_num_layers=2, dropout=0.5):
        super(LipNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        self.rnn_input_size = 96 * 5 * 5
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        
        x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.size()
        x = x.reshape(B, T, -1)

        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)

        x = self.fc_layers(x[:, -1, :])

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)