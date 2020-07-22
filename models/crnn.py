import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        in_channels = 3
        out_channels = 32
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(1)

        hidden_num = 128
        self.rnn = nn.LSTM(55, hidden_num, 1)
        self.linear = nn.Linear(hidden_num, 2)

    def forward(self, x):
        # [32, 3, 224, 224]
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 3, stride=2, padding=1)
        # [32, 32, 111, 111]
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 3, stride=2, padding=1)
        # [32, 1, 55, 55]
        out = out.squeeze()
        # [32, 55, 55]
        out, _ = self.rnn(out)
        # [32, 55, 32]
        out = out[:, -1, :]
        # [32, 32]
        out = self.linear(out)
        # [32, 2]
        return out
