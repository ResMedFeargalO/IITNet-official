import torch
import torch.nn as nn
from torch.nn import Conv1d, MaxPool1d,Conv2d


class DeepSleepNetFeature(nn.Module):
    def __init__(self, config, dropout=0.5):
        super(DeepSleepNetFeature, self).__init__()

        self.chn = 64

        # architecture
        self.config = config
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.path1 = nn.Sequential(Conv1d(1, self.chn, 50, 6, bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(),
                                   MaxPool1d(8),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn*2, 8, 1, bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, bias=False),
                                   nn.BatchNorm1d(self.chn * 2),
                                   nn.ReLU(),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(),
                                   MaxPool1d(4)
                                   )
        self.path2 = nn.Sequential(Conv1d(1, self.chn, 400, 50, bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(),
                                   MaxPool1d(4),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn*2, 8, 1, bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, bias=False),
                                   nn.BatchNorm1d(self.chn * 2),
                                   nn.ReLU(),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(),
                                   MaxPool1d(2))
        
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
        x2 = self.path2(x)  # path 2

        if self.config['n_anchor'] == 1:
            x2 = torch.nn.functional.interpolate(x2, x1.size(2))
            x = self.dropout(self.smooth(self.compress(torch.cat([x1, x2], dim=1))))

            return x
        
        elif self.config['n_anchor'] == 2:

            return [x1, x2]
        
        else:
            raise NotImplementedError