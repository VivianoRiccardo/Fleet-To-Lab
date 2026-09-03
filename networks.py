import torch
import torch.nn as nn


class AdaptiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return self.linear(x) * self.gamma + self.beta


class AdaptiveSlippageNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        '''
        self.layer1 = AdaptiveLinear(input_size, 128)
        self.norm1 = nn.LayerNorm(128)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)

        self.layer2 = AdaptiveLinear(128, 64)
        self.norm2 = nn.LayerNorm(64)
        self.act2 = nn.GELU()

        self.layer3 = AdaptiveLinear(64, 32)
        self.norm3 = nn.LayerNorm(32)
        self.act3 = nn.GELU()

        self.output_layer = AdaptiveLinear(32, num_classes)
        '''
        
        self.layer1 = AdaptiveLinear(input_size, 30)
        self.norm1 = nn.LayerNorm(30)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)

        self.output_layer = AdaptiveLinear(30, num_classes)
        
    def forward(self, x):
        '''
        x = self.dropout1(self.act1(self.norm1(self.layer1(x))))
        x = self.act2(self.norm2(self.layer2(x)))
        x = self.act3(self.norm3(self.layer3(x)))
        return self.output_layer(x)
        '''
        
        return self.output_layer(self.dropout1(self.act1(self.norm1(self.layer1(x)))))
        
    def set_mode(self, mode="base"):
        for n, p in self.named_parameters():
            if mode == "base":
                p.requires_grad = not ("gamma" in n or "beta" in n)
            else:
                p.requires_grad = ("gamma" in n or "beta" in n)


class SlippageNet(nn.Module):
    def __init__(self, input_size=12, num_classes=3):
        super().__init__()
        
        '''
        self.layer1 = nn.Linear(input_size, 128)
        self.norm1 = nn.LayerNorm(128)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)

        self.layer2 = nn.Linear(128, 64)
        self.norm2 = nn.LayerNorm(64)
        self.act2 = nn.GELU()

        self.layer3 = nn.Linear(64, 32)
        self.norm3 = nn.LayerNorm(32)
        self.act3 = nn.GELU()

        self.output_layer = nn.Linear(32, num_classes)
        '''
        
        self.layer1 = nn.Linear(input_size, 30)
        self.norm1 = nn.LayerNorm(30)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)
        self.output_layer = nn.Linear(30, num_classes)
        
    def forward(self, x):
        '''
        x = self.dropout1(self.act1(self.norm1(self.layer1(x))))
        x = self.act2(self.norm2(self.layer2(x)))
        x = self.act3(self.norm3(self.layer3(x)))
        return self.output_layer(x)
        '''
        return self.output_layer(self.dropout1(self.act1(self.norm1(self.layer1(x)))))
        
