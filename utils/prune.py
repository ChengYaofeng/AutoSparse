from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def prune_loop(model, loss, pruner, dataloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False, 
               params = None, rewind=True, prediction_model= None): #这里的命名感觉不是很好
    '''
    Applies score mask loop iteratively to a final sparsity level.
    Input:
        model:
        loss:
        pruner: {class} pruner, SNIP, GraSP, SynFlow, etc.

    '''
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()
        
    # Prune model
    for epoch in tqdm(range(epochs)):  #执行1次，之前synflow用
        
        pruner.init_p_grad(model, loss, dataloader, device)
        
        if (pruner.__class__.__name__ == 'DynamicIntegratedGradients'):
            pruner.score(model, loss, dataloader, device, sparsity)
            
        elif prediction_model is not None:
            prediction_model.eval()
            with torch.no_grad():
                for i, (_, p0) in enumerate(pruner.masked_params): #如何理解p0
                    p = pruner.dict['param'][i].reshape(-1)
                    g = pruner.dict['grad'][i].reshape(-1)
                    dataset = TensorDataset(p, g)
                    important_list = torch.tensor([]).cpu()
                    dataloader = DataLoader(dataset, batch_size=1024*8, shuffle=False)     # batchsize需要进行调整吗？

                    for batch_p, batch_g in dataloader:
                        batch_p, batch_g = batch_p.to(device), batch_g.to(device)
                        output = prediction_model(batch_p, batch_g)
                        output = output.squeeze(-1)
                        important_list = torch.cat((important_list, torch.clone(output).detach().cpu()), dim=0) # 这里不放到cpu上可以吗？ 显存
                    
                    #连接所有结果并reshape为p0
                    important = important_list.view(p0.shape)
                    pruner.scores[id(p0)] = important.to(device)      #这里是scores，是参数，score是函数
        
        else:
            pruner.score(model, loss, dataloader, device)
            
            
        # invert scores 在for循环里面还是外面
        if invert:
            pruner.invert()
        
        pruner.mask(sparsity, scope, rewind)   #这里的scope是
        
    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()
        
    # Confirm sparsity level 这里的作用是什么
    remaining_params, total_params = pruner.stats()