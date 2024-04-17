import torch
import torch.nn as nn

from models import tinyimagenet_resnet, tinyimagenet_vgg
from models.transformer import KNOWN_MODELS

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(2, 16),
        nn.BatchNorm1d(16), 
        nn.ReLU(), 
        nn.Linear(16, 64),
        nn.BatchNorm1d(64), 
        nn.ReLU(),                                
        nn.Linear(64, 1024),
        nn.BatchNorm1d(1024),  
        nn.ReLU(),     
        # nn.Linear(1024, 1024),
        # nn.BatchNorm1d(1024),  
        # nn.ReLU(), 
        # nn.Linear(1024, 1024),
        # nn.BatchNorm1d(1024),  
        # nn.ReLU(), 
        # nn.Linear(1024, 1024),
        # nn.BatchNorm1d(1024),  
        nn.ReLU(), 
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),  
        nn.ReLU(), 
        nn.Linear(1024, 64),
        nn.BatchNorm1d(64),  
        nn.ReLU(),                       
        nn.Linear(64, 1)     
        
    ) 
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    

    def forward(self, params, grads):
        combined = torch.cat((params.unsqueeze(-1), grads.unsqueeze(-1)), dim=1)
        # print(combined.shape)
        output = self.model(combined)
        return output
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64*16*3),
            nn.BatchNorm1d(64*16*3),
            nn.ReLU(),
        )

        self.resnet18 = tinyimagenet_resnet.resnet18(32,1)

    def forward(self, params, grads):
        combined = torch.cat((params.unsqueeze(-1), grads.unsqueeze(-1)), dim=1)
        mlp_output = self.mlp(combined).reshape(-1, 3, 32, 32)  # 修改此处
        return self.resnet18(mlp_output)
    
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64*16*3),
            nn.BatchNorm1d(64*16*3),
            nn.ReLU(),
        )

        self.resnet18 = tinyimagenet_resnet.resnet50(32,1)

    def forward(self, params, grads):
        combined = torch.cat((params.unsqueeze(-1), grads.unsqueeze(-1)), dim=1)
        mlp_output = self.mlp(combined).reshape(-1, 3, 32, 32)  # 修改此处
        return self.resnet18(mlp_output)

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64*64),
            nn.BatchNorm1d(64*64),
            nn.ReLU(),
            nn.Linear(64*64, 64*64*3),
            nn.BatchNorm1d(64*64*3),
            nn.ReLU(),
        )

        self.vgg19 = tinyimagenet_vgg.vgg19_bn(64,1)

    def forward(self, params, grads):
        combined = torch.cat((params.unsqueeze(-1), grads.unsqueeze(-1)), dim=1)
        
        mlp_output = self.mlp(combined).reshape(-1, 3, 64, 64)  # 修改此处
        return self.vgg19(mlp_output)

class BiT(nn.Module):
    def __init__(self):
        super(BiT, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32*32),
            nn.BatchNorm1d(32*32),
            nn.ReLU(),
            nn.Linear(32*32, 32*32*3),
            nn.BatchNorm1d(32*32*3),
            nn.ReLU(),
        )

        self.bit = KNOWN_MODELS['BiT-M-R152x2'](head_size=1, zero_head=True)


    def forward(self, params, grads):
        combined = torch.cat((params.unsqueeze(-1), grads.unsqueeze(-1)), dim=1)
        
        mlp_output = self.mlp(combined).reshape(-1, 3, 32, 32)  # 修改此处
        return self.bit(mlp_output)