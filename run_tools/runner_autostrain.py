import argparse
import time
import os
import torch
import pickle
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from autos_model.autosnet import MLP, ResNet18, Vgg19, BiT
from run_utils.logger import get_root_logger, print_log
from utils.data_visual import predict_error_visual

def run(cfgs):
    
    train_cfgs = cfgs['train']
    policy_cfgs = cfgs['policy']
    
    # path
    timestamp = time.localtime()
    formatted_time = time.strftime("%Y%m%d-%H-%M", timestamp)
    root = os.getcwd()
    data_load_path = os.path.join(root, 'dumps', train_cfgs['model'], train_cfgs['dataset'], train_cfgs['method'], train_cfgs['save_name'], 'data.pkl')
    prediction_model_path = os.path.join(root,'prediction_model', policy_cfgs['prediction_model'], train_cfgs['model'], train_cfgs['dataset'], 
                                         train_cfgs['method'], train_cfgs['save_name'], formatted_time)
    if not os.path.exists(prediction_model_path):
        os.makedirs(prediction_model_path)
        
    device = torch.device(("cuda:" + str(policy_cfgs['gpu'])) if torch.cuda.is_available() else "cpu")
    
    # log & save path
    logger = get_root_logger(os.path.join(prediction_model_path, 'log.log'), name='autos_network')
    save_dict = {}
    save_dict['train_loss'] = []
    save_dict['test_loss'] = []
    
    
    # data
    with open(data_load_path, 'rb') as f:
        train_data = pickle.load(f)
        
    params = torch.cat([p.reshape(-1) for p in train_data['params']])   # traindata 在哪里训练
    grads = torch.cat([g.reshape(-1) for g in train_data['grads']])
    importants = torch.cat([imp.reshape(-1) for imp in train_data['importants']])
    
    dataset = TensorDataset(params, grads, importants)
    train_dataloader = DataLoader(dataset, batch_size=policy_cfgs['batch_size'], shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=policy_cfgs['batch_size'], shuffle=True)
    
    
    # model
    if policy_cfgs['prediction_model'] == "fc":
        model = MLP().to(device)
    elif policy_cfgs['prediction_model'] == "resnet18":
        model = ResNet18().to(device)
    elif policy_cfgs['prediction_model'] == "vgg19":
        model = Vgg19().to(device)
    elif policy_cfgs['prediction_model'] == "bit":
        model = BiT().to(device)
    else:
        raise ValueError("model not found")
    
    # 这里考虑在其他地方？
    # for arg in vars(args):
    #     print_log(f"{arg}: {getattr(args, arg)}", logger=logger)
        
    # loss, optimizer
    loss_cal = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=policy_cfgs['lr'])
    
    total_loss = 0.
    all_loss = []
    
    # 显示一下训练前后的差距
    
    # train
    for epoch in range(policy_cfgs['epochs']):
        # print(policy_cfgs['epochs'])
        for i,(batch_params, batch_grads, batch_importants) in enumerate(train_dataloader):
            batch_params, batch_grads, batch_importants = batch_params.to(device), batch_grads.to(device), batch_importants.to(device)
            optimizer.zero_grad()
            
            output = model(batch_params, batch_grads).squeeze(-1)
            # print(batch_importants.shape)
            loss = loss_cal(output, batch_importants)
            loss.backward()
            optimizer.step()
        all_loss.append(loss.item())
        # print_log(f'Train Epoch [{epoch + 1}/{policy_cfgs['epochs']}], Loss: {loss.item():.4f}', logger=logger)
    
    save_dict['train_loss'] = all_loss
    
    
    # test
    model.eval()
    with torch.no_grad():
        for _, (batch_params, batch_grads, batch_importants) in enumerate(test_dataloader):
            batch_params, batch_grads, batch_importants = batch_params.to(device), batch_grads.to(device), batch_importants.to(device)
            output = model(batch_params, batch_grads)
            output.squeeze_(-1)
            
            loss = loss_cal(output, batch_importants)
            total_loss += loss.item() * len(batch_params)  # 这里什么要乘长度
        
            avg_loss = total_loss / len(test_dataloader)
            
            predict_error_visual(batch_params, batch_grads, batch_importants, output, os.path.join(prediction_model_path, 'after'))
    
    save_dict['test_loss'] = avg_loss
    
    torch.save(save_dict, os.path.join(prediction_model_path, 'result.pth'))   #后面画图loss
    torch.save(model, os.path.join(prediction_model_path, 'model.pth'))

