import torch
import numpy as np
from torch.nn import functional as F
from time import time
from torch.utils.data import DataLoader, TensorDataset


class Pruner:
    def __init__(self, masked_params):
        '''
        base prune class
        Input:
            masked_params: {generator data(Tensor)} masks and parameters
        '''
        self.masked_params = list(masked_params)
        self.scores = {}
        self.dict = {}
        self.dict['importants'] = [torch.zeros_like(mask) for mask, _ in self.masked_params]
        self.dict['params'] = [torch.clone(p.data).detach() for (_, p) in self.masked_params]   # 这里不加括号会不会有问题，这里不能加s
        self.x = 0            # 这个参数干啥的
        
        
    def score(self, model, loss, dataloader, device, autos_model=None):
        raise NotImplementedError
        

    def _global_mask(self, sparsity, rewind):
        '''
        Updates masks of model with scores by sparsity level globally.
        Input:
            sparsity: {}, sparsity level
            rewind: {bool}, whether to rewind to initial weights
        '''
        
        # threshold scores
        global_scores = torch.cat([torch.flatten(torch.masked_select(v, v != 0)) for v in self.scores.values()])
        
        sparse_k = int((1 - sparsity) * global_scores.numel())
        
        if not sparse_k < 1:
            threshold, _ = torch.kthvalue(global_scores, sparse_k)
            for idx, (mask, param) in enumerate(self.masked_params):
                score = self.scores[id(param)]
                zero = torch.zeros_like(mask, device=mask.device)
                one = torch.ones_like(mask, device=mask.device)
                mask.data = (torch.where(score <= threshold, zero, one))
                if rewind == True:
                    param.data = (torch.where(score <= threshold, zero, self.dict['params'][idx]))
        
        self.apply_mask()
                
    
    @torch.no_grad()
    def apply_mask(self):
        '''
        Applies mask to prunable parameters.
        '''        
        for mask, param in self.masked_params:
            param.mul_(mask)
        
        
    def mask(self, sparsity, scope, rewind):
        '''
        Updates masks of model with scores by sparsity according to scope
        '''
        if scope == 'global':
            self._global_mask(sparsity, rewind)
        if scope == 'local':
            self._local_mask(sparsity)
        self.important()
        
    
    def stats(self):
        '''
        Returns remaining and total number of prunable parameters. #detach().cpu().numpy().
        '''
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_params:
            remaining_params += mask.sum()
            total_params += mask.numel()
        return remaining_params, total_params
        
    
    def important(self):
        '''
        Updates important scores
        '''
        for idx, (mask, _) in enumerate(self.masked_params):
            self.dict['importants'][idx] += torch.clone(mask.data).detach()
    
    
    def init_p_grad(self, model, loss, dataloader, device):
        '''
        Initializes gradients of masked parameters.
        '''
        if self.x == 0:
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss(output, target).backward()
            
            self.dict['grads'] = [torch.clone(p.grad.data).detach() for (_, p) in self.masked_params]   #这里之前的数据集是grad
            print('gradient init')
        self.x += 1
        # print(f'x: {self.x}')


class Random(Pruner):
    def __init__(self, masked_params):
        super(Random, self).__init__(masked_params)
        
    def score(self, model, loss, dataloader, device, autos_model=None):
        for _, p in self.masked_params:
            self.scores[id(p)] = torch.randn_like(p)
            
            
class Magnitude(Pruner):
    def __init__(self, masked_params):
        super(Magnitude, self).__init__(masked_params)
    
    def score(self, model, loss, dataloader, device, autos_model=None):
        for _, p in self.masked_params:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()  #这里abs_()是原地操作，可以节省内存
            

class SNIP(Pruner):
    def __init__(self, masked_params):
        super(SNIP, self).__init__(masked_params)
        
    def score(self, model, loss, dataloader, device, autos_model=None):
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
        


class AutoS(Pruner):
    def __init__(self, masked_params, params_batch_size=4096):
        super(AutoS, self).__init__(masked_params)
        self.params_batch_size = params_batch_size
        
    def score(self, model, loss, dataloader, device, autos_model):
        '''
        Input:
             
        '''
        # compute grad 存疑
        # for batch_idx, (data, target) in enumerate(dataloader):
        #         data, target = data.to(device), target.to(device)
        #         output = model(data)
        #         loss(output, target).backward()
        
        # self.dict['grads'] = [torch.clone(p.grad.data).detach() for (_, p) in self.masked_params]
        self.init_p_grad(model, loss, dataloader, device)
        
        autos_model.eval()
        with torch.no_grad():
            for i, (_, p0) in enumerate(self.masked_params): #如何理解p0
                # print(self.dict.keys())
                print(i)
                p = self.dict['params'][i].reshape(-1)
                g = self.dict['grads'][i].reshape(-1)
                dataset = TensorDataset(p, g)
                important_list = torch.tensor([]).cpu()
                dataloader = DataLoader(dataset, batch_size=self.params_batch_size, shuffle=False)     # batchsize需要进行调整吗？

                for batch_p, batch_g in dataloader:
                    batch_p, batch_g = batch_p.to(device), batch_g.to(device)
                    output = autos_model(batch_p, batch_g)
                    output = output.squeeze(-1)
                    important_list = torch.cat((important_list, torch.clone(output).detach().cpu()), dim=0) # 这里不放到cpu上可以吗？ 显存
                
                #连接所有结果并reshape为p0
                important = important_list.view(p0.shape)
                self.scores[id(p0)] = important.to(device)      #这里是scores，是参数，score是函数
        del autos_model
