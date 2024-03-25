from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def prune_loop(model, loss, pruner, dataloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False, 
               params = None, prediction_model= None, rewind=True):
    '''
    Applies score mask loop iteratively to a final sparsity level.
    Input:

    '''
    