import time
import os
import torch
import torch.nn as nn
import pickle
import pandas as pd

from run_utils.logger import get_root_logger, print_log
from utils import loader
from utils import generator, prune, train, metrics


def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run(cfgs):
    '''
    Input:
        cfgs: {dict}, config params
    '''
    
    # steps = 0 #后面看看在哪里单独指定  10
   
    # cfgs
    train_cfgs = cfgs['train']
    prune_cfgs = cfgs['prune']
    policy_cfgs = cfgs['policy']
   
    # seed & device
    torch.manual_seed(policy_cfgs['seed'])
    torch.cuda.manual_seed(policy_cfgs['seed'])
    
    device = torch.device(("cuda:" + str(policy_cfgs['gpu'])) if torch.cuda.is_available() else "cpu")
    
    # log & save path
    timestamp = time.localtime()
    formatted_time = time.strftime("%Y%m%d-%H%M", timestamp)
    file_name = f"{policy_cfgs['expid']}-{formatted_time}"
    result_dir = policy_cfgs['result_dir']
    dataset_path = f"autos_dataset/{train_cfgs['model']}/{train_cfgs['dataset']}/{prune_cfgs['pruner']}/{file_name}"
    result_path = f"{os.getcwd()}/{result_dir}"
    save_path = f"{os.getcwd()}/{result_dir}/save/"   # 这里直接放到prune_result里面好吗
    checkdir(result_path)
    checkdir(save_path)
    
    log_file = os.path.join(f'{result_path}/{file_name}.log') 
    logger = get_root_logger(log_file, name='singleshot')
    
    # data
    print_log('Loading {} dataset.'.format(train_cfgs['dataset']), logger=logger)
    input_shape, num_classes = loader.dimension(train_cfgs['dataset'])
    
    prune_loader = loader.dataloader(train_cfgs['dataset'], train_cfgs['train_batchsize'], True, policy_cfgs['workers'])
    train_loader = loader.dataloader(train_cfgs['dataset'], train_cfgs['train_batchsize'], True, policy_cfgs['workers'])
    test_loader = loader.dataloader(train_cfgs['dataset'], train_cfgs['test_batchsize'], False, policy_cfgs['workers'])
    
    # model, optimizer, criterion
    print_log('Creating {}-{} model.'.format(train_cfgs['model_class'], train_cfgs['model']), logger=logger)
    model = loader.model(train_cfgs['model'], train_cfgs['model_class'])(input_shape, 
                                                     num_classes, 
                                                     train_cfgs['dense_classifier'], 
                                                     train_cfgs['pretrained']).to(device)
    
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = loader.optimizer(train_cfgs['optimizer'])
    optimizer = opt_class(generator.parameters(model), lr=train_cfgs['lr'], weight_decay=train_cfgs['weight_decay'], **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_cfgs['lr_drops'], gamma=train_cfgs['lr_drop_rate'])
    
    # pretrain 这里的意思是什么
    print_log('Pre-Train for {} epochs.'.format(train_cfgs['pre_epochs']),logger=logger)
    pre_result = train.train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                 test_loader, device, train_cfgs['pre_epochs'], policy_cfgs['verbose'])
    
    # prune
    print_log('Pruning with {} for {} epochs.'.format(prune_cfgs['pruner'], prune_cfgs['prune_epochs']),logger=logger)
    masked_params = generator.masked_parameters(model, prune_cfgs['prune_bias'], prune_cfgs['prune_batchnorm'], prune_cfgs['prune_residual']) #mask & params
    pruner = loader.pruner(prune_cfgs['pruner'])(masked_params)
    
    if policy_cfgs['run_choice'] == 'prune_iterative': #'train_important','prune_once','prediction_prune' 这里的分类是？迭代剪枝，单次剪枝，预测剪枝
        sparse  = prune_cfgs['compression']
        prediction_model = None
        # if steps != 0:
        #     with open(cfgs['save_important'], 'rb') as f:
        #         data = pickle.load(f)
        #     #dataset融合
        #     for idx, (_, p) in enumerate(pruner.masked_params):              # 这个循环没有看懂
        #         p = data['param'][idx].to(device)
        #         pruner.dict['importants'][idx] = data['importants'][idx].to(device)     #这里放的是？
        # else:
        #     checkdir(f"{os.getcwd()}/{dataset_path}")
        #     torch.save(model.state_dict(),"{}/before_train_model.pt".format(f"{os.getcwd()}/{dataset_path}"))
        #     cfgs['save_important'] = f"{os.getcwd()}/{dataset_path}/data.pkl"  # 更新了命令行参数
        dataset_path = f"{os.getcwd()}/{result_dir}/dataset/"
        checkdir(dataset_path)
        torch.save(model.state_dict(),"{}/before_train_model.pt".format(dataset_path))
        policy_cfgs['save_important'] = f"{dataset_path}/data.pkl"  # 更新了命令行参数
            
    elif policy_cfgs['run_choice'] == 'prune_once':
        sparse = policy_cfgs['singleshot_compression'][policy_cfgs['times']]
        prediction_model = None
        prune_cfgs['prune_epochs'] = 1
        
    elif policy_cfgs['run_choice'] == 'prune_autos':
        sparse = policy_cfgs['singleshot_compression'][policy_cfgs['times']]
        prune_cfgs['prune_epochs'] = 1
        if policy_cfgs['autos_model'] != None:
            prediction_model = torch.load(policy_cfgs['autos_model'], map_location=torch.device(device)).to(device)
        else:
            raise ValueError("No autos_model is given")
    
    
    for i in range(prune_cfgs['prune_epochs']):
        print_log(f'----------prune epoch{i}----------', logger=logger)
        
        # 定义迭代稀疏的剪枝多少
        sparsity = 1 - (1 - sparse) / (prune_cfgs['prune_epochs'] - (1 - sparse) * i)
        # 稀疏的比例，上面的是按照特定的数量稀疏
        
        # 15s
        #start_time = time.time()
        prune.prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
                prune_cfgs['compression_schedule'], prune_cfgs['mask_scope'], 1, prune_cfgs['reinitialize'], prune_cfgs['prune_train_mode'], 
                prune_cfgs['shuffle'], prune_cfgs['invert'], prune_cfgs['rewind'], prediction_model=prediction_model, choice=policy_cfgs['run_choice'])
        # print_log(f"Pruning time:{'-'*20} {time.time() - start_time}", logger=logger)
        
        optimizer = opt_class(generator.parameters(model), lr=train_cfgs['lr'], weight_decay=train_cfgs['weight_decay'], **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_cfgs['lr_drops'], gamma=train_cfgs['lr_drop_rate'])
        
    
        # post train
        print_log('Post-Training for {} epochs.'.format(train_cfgs['post_epochs']),logger=logger)
        post_result = train.train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                        test_loader, device, train_cfgs['post_epochs'], policy_cfgs['verbose'])
        
        # results
        max_accuracy1_row_index = post_result['top1_accuracy'].idxmax()
        max_accuracy1_row = post_result.iloc[max_accuracy1_row_index].to_frame().T
        
        frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), max_accuracy1_row]#post_result.tail(1)
        train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
        prune_result = metrics.summary(model, 
                                    pruner.scores,
                                    metrics.flop(model, input_shape, device),
                                    lambda p: generator.prunable(p, prune_cfgs['prune_batchnorm'], prune_cfgs['prune_residual']))
        total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
        possible_params = prune_result['size'].sum()
        total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
        possible_flops = prune_result['flops'].sum()
        print_log(f"Train results:\n {train_result}", logger=logger)
        print_log(f"Prune results:\n {prune_result}", logger=logger)
        print_log("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params), logger=logger)
        print_log("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops), logger=logger)

    # save results and model
    if policy_cfgs['save_important'] is not None:
        for key, tensor_list in pruner.dict.items():
            pruner.dict[key] = [tensor.cpu() for tensor in tensor_list]
        checkdir(f"{os.getcwd()}/{dataset_path}/")
        with open(policy_cfgs['save_important'], 'wb') as fp:
            # print(pruner.dict)
            pickle.dump(pruner.dict, fp)
        print_log(f"data saved at {policy_cfgs['save_important']}", logger=logger)
    
    
    if policy_cfgs['save']:
        print_log(f"Saving results at{save_path}", logger=logger)
        pre_result.to_pickle("{}/pre-train.pkl".format(save_path))
        post_result.to_pickle("{}/post-train.pkl".format(save_path))
        prune_result.to_pickle("{}/compression.pkl".format(save_path))
        torch.save(model.state_dict(),"{}/model.pt".format(save_path))
        torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(save_path))
        torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(save_path))