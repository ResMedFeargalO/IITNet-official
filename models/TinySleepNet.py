import torch
import torch.nn as nn
from torch.nn import Conv1d, MaxPool1d,Conv2d


class TinySleepNetFeature(nn.Module):
    def __init__(self, config, dropout=0.5):
        super(TinySleepNetFeature, self).__init__()

        self.chn = 128

        # architecture
        self.config = config
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.path1 = nn.Sequential(Conv1d(1, self.chn, 50, 6, bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(),
                                   MaxPool1d(8,stride=8),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn, 8, 1, padding='same', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(),
                                   Conv1d(self.chn, self.chn, 8, 1, padding='same', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(),
                                   Conv1d(self.chn, self.chn, 8, 1, padding='same', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(),
                                   MaxPool1d(4, stride=4)
                                   )
        
        if config['init_weights']:
            self._initialize_weights()
        
        if config['n_anchor'] == 1:
            self.compress = nn.Conv1d(self.chn * 4, self.chn * 2, 1, 1, 0)
            self.smooth = nn.Conv1d(self.chn * 2, self.chn * 2, 3, 1, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.path1(x)  # path 1
        
        return x1