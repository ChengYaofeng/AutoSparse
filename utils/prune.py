from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def prune_loop(model, loss, pruner, pruneloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False, 
               params = None, rewind=True, prediction_model= None, choice=None): #这里的命名感觉不是很好
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
        
        if choice == 'prune_iterative':
            pruner.init_p_grad(model, loss, pruneloader, device)
            pruner.score(model, loss, pruneloader, device)
        else:
            pruner.score(model, loss, pruneloader, device, prediction_model)
            
            
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
    