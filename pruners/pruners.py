import torch
import numpy as np
from torch.nn import functional as F

class Pruner:
    def __init__(self, masked_params):
        self.masked_params = list(masked_params)
        self.scores = {}
        self.dict = {}
        self.dict['importants'] = [torch.zeros_like(mask) for mask, _ in self.masked_params]
        self.dict['params'] = [torch.clone(p.data).detach() for (_, p) in self.masked_params]   # 这里不加括号会不会有问题
        self.x = 0            # 这个参数干啥的
        
        
    def score(self, model, loss, dataloader, device):
        raise NotImplementedError
        

    def _global_mask(self, sparsity, rewind):
        '''
        Updates masks of model with scores by sparsity level globally.
        Input:
            sparsity: {}, sparsity level
            rewind: {bool}, whether to rewind to initial weights
        '''
        
        1
        


class Random(Pruner):
    def __init__(self, masked_params):
        super(Random).__init__(masked_params)
        
    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_params:
            self.scores[id(p)] = torch.randn_like(p)
            
            
class Magnitude(Pruner):
    def __init__(self, masked_params):
        super(Magnitude).__init__(masked_params)
    
    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_params:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()  #这里abs_()是原地操作，可以节省内存
            

class SNIP(Pruner):
    def __init__(self, masked_params):
        super(SNIP, self).__init__(masked_params)
        
    def score(self, model, loss, dataloader, device):
        '''
        Input:
            loss: {function}
            model:
            dataloader: 
        '''
        
        # allow mask to have gradient
        for m, _ in self.masked_params:
            m.requires_grad = True
            
        # compute gradients
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()
            
        # compute scores |g * theta|
        for m, p in self.masked_params:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False
            
        # normalize scores
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_params:
            self.scores[id(p)].div_(norm)          #针对每个元素进行操作
            

class GraSP(Pruner):
    def __init__(self, masked_params):
        super(GraSP, self).__init__(masked_params)
        




