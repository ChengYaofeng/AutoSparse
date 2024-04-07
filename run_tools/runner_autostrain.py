import argparse
import time
import os
import torch
import pickle
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from autos_model.autosnet import MLP, ResNet18, Vgg19, BiT
from run_utils.logger import get_root_logger, print_log
from utils.data_visual import predict_error_visual


def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def run(cfgs):
    
    train_cfgs = cfgs['train']
    prune_cfgs = cfgs['prune']
    policy_cfgs = cfgs['policy']
    
    # path
    timestamp = time.localtime()
    formatted_time = time.strftime("%Y%m%d-%H-%M", timestamp)
    file_name = f"{policy_cfgs['expid']}-{formatted_time}"
    root = os.getcwd()
    result_dir = policy_cfgs['result_dir']
    dataset_path = train_cfgs['dataset_path']
    data_load_path = os.path.join(root, dataset_path)#, 'data.pkl')

    log_path = os.path.join(root, result_dir)
    model_save_path = os.path.join(root, result_dir, file_name)
    checkdir(log_path)
    checkdir(model_save_path)
        
    device = torch.device(("cuda:" + str(policy_cfgs['gpu'])) if torch.cuda.is_available() else "cpu")
    
    # log & save path
    log_file = os.path.join(f'{log_path}/{file_name}.log')
    logger = get_root_logger(log_file, name='autos_network')
    save_dict = {}
    save_dict['train_loss'] = []
    save_dict['test_loss'] = []
    
    
    # data
    print_log(f"Loading data from {data_load_path}", logger=logger)
    with open(data_load_path, 'rb') as f:
        train_data = pickle.load(f)
        
    params = torch.cat([p.reshape(-1) for p in train_data['param']])
    grads = torch.cat([g.reshape(-1) for g in train_data['grad']])
    importants = torch.cat([imp.reshape(-1) for imp in train_data['importants']])
    
    max_imp = importants.max()
    min_imp = importants.min()
    importants = (importants - min_imp + 1e-7) / (max_imp - min_imp + 1e-7)  # N
    
    dataset = TensorDataset(params, grads, importants)
    train_dataloader = DataLoader(dataset, batch_size=train_cfgs['train_batchsize'], shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=train_cfgs['test_batchsize'], shuffle=True)
    
    
    # model
    print_log(f"Training model is {train_cfgs['prediction_model']}", logger=logger)
    if train_cfgs['prediction_model'] == "fc":
        model = MLP().to(device)
    elif train_cfgs['prediction_model'] == "resnet18":
        model = ResNet18().to(device)
    elif train_cfgs['prediction_model'] == "vgg19":
        model = Vgg19().to(device)
    elif train_cfgs['prediction_model'] == "bit":
        model = BiT().to(device)
    else:
        raise ValueError("model not found")
    
    # 这里考虑在其他地方？
    # for arg in vars(args):
    #     print_log(f"{arg}: {getattr(args, arg)}", logger=logger)
        
    # loss, optimizer
    loss_cal = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfgs['lr'])
    
    total_loss = 0.
    all_loss = []
    
    # 显示一下训练前后的差距
    log_steps = len(train_dataloader) // 10
    
    # train
    for epoch in range(train_cfgs['epochs']):
        print_log(f'-----------------Pretrain epoch {epoch}-----------------', logger=logger)
        for batch_idx, (batch_params, batch_grads, batch_importants) in enumerate(tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9)):
            
            batch_params, batch_grads, batch_importants = batch_params.to(device), batch_grads.to(device), batch_importants.to(device)
            optimizer.zero_grad()
            
            output = model(batch_params, batch_grads).squeeze(-1)
            # print(batch_importants.shape)
            loss = loss_cal(output, batch_importants)
            loss.backward()
            optimizer.step()
            if (batch_idx) % log_steps == 0:
                print_log(f'Train Epoch [{epoch + 1}/{train_cfgs["epochs"]}], Loss: {loss.item():.4f}', logger=logger)
        all_loss.append(loss.item())
    
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
                
                predict_error_visual(batch_params, batch_grads, batch_importants, output, os.path.join(model_save_path, 'after'))
        
        save_dict['test_loss'] = avg_loss
        
        torch.save(save_dict, os.path.join(model_save_path, 'result.pth'))   #后面画图loss
        torch.save(model, os.path.join(model_save_path, 'model.pth'))

