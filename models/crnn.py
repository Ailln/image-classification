import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        in_channels = 3
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=3, stride=3, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        hidden_num = 32
        self.rnn = nn.LSTM(74, hidden_num, 1)
        self.linear = nn.Linear(hidden_num, 2)

    def forward(self, x):
        # [64, 3, 224, 224]
        out = F.relu(self.bn1(self.conv1(x)))
        out = out.squeeze()
        out, _ = self.rnn(out)
        out = out[:, -1, :]
        # [32, 32]
        out = self.linear(out)
        # [32, 2]
        return out
