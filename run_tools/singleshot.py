import time
import os
import torch
import torch.nn as nn
import pickle

from run_utils.logger import get_root_logger, print_log
from run_utils import loader, trainer
from utils import generator


def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run(args, steps):
    '''
    Input:
        args: {argparse}, arguments
        steps: {int}, 这里是循环的步数，是不是能放到函数里面
    '''
   
    # seed & device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    device = torch.device(("cuda:" + str(args.gpu)) if torch.cuda.is_available() else "cpu")
    
    # log & save path
    timestamp = time.localtime()
    formatted_time = time.strftime("%Y%m%d-%H-%M", timestamp)
    file_name = f'{args.expid}-{formatted_time}'              # 这里有点啰嗦
    log_path = f"{os.getcwd()}/log/{args.model}/{args.dataset}/{args.run_choice}/{file_name}/"
    save_path = f"{os.getcwd()}/prune_result/{args.model}/{args.dataset}/{args.run_choice}/{file_name}/"   # 这里直接放到prune_result里面好吗
    checkdir(log_path)
    checkdir(save_path)
    
    log_file = os.path.join(f'{args.model}/{args.dataset}/{args.run_choice}/{file_name}.log')
    
    logger = get_root_logger(log_file, name='singleshot')
    
    # data
    print_log('Loading {} dataset.'.format(args.dataset), logger=logger)
    input_shape, num_classes = loader.dimension(args.dataset)
    
    prune_loader = loader.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    train_loader = loader.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = loader.dataloader(args.dataset, args.test_batch_size, False, args.workers)
    
    # model, optimizer, criterion
    print_log('Creating {}-{} model.'.format(args.model_class, args.model), logger=logger)
    model = loader.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained).to(device)
    
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = loader.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)
    
    # pretrain 这里的意思是什么
    print_log('Pre-Train for {} epochs.'.format(args.pre_epochs),logger=logger)
    pre_result = trainer.train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                 test_loader, device, args.pre_epochs, args.verbose)
    
    # prune
    print_log('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs),logger=logger)
    pruner = loader.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
    
    if args.run_choice == 'prune_iterative':#'train_important','prune_once','prediction_prune' 这里的分类是？迭代剪枝，单次剪枝，预测剪枝
        sparse  = 0.01
        prediction_model = None
        args.prune_epochs = 20
        if steps != 0:
            with open(args.save_important, 'rb') as f:
                data = pickle.load(f)
            for idx, (_, p) in enumerate(pruner.masked_params):              # 这个循环没有看懂
                p = data['param'][idx].to(device)
                pruner.dict['importants'][idx] = data['importants'][idx].to(device)     #这里放的是？
        else:
            checkdir(f"{os.getcwd()}/dumps/{args.model}/{args.dataset}/{args.pruner}/{file_name}/")
            torch.save(model.state_dict(),"{}/before_train_model.pt".format(f"{os.getcwd()}/dumps/{args.model}/{args.dataset}/{args.pruner}/{file_name}/"))
            args.save_important = f"{os.getcwd()}/dumps/{args.model}/{args.dataset}/{args.pruner}/{file_name}/data.pkl"  # 更新了命令行参数
            
    elif args.run_choice == 'prune_once':
        sparse = args.singleshot_compression[steps]
        prediction_model = None
        args.prune_epochs = 1
        
    elif args.run_choice == 'prune_prediction':
        sparse = args.singleshot_compression[steps]
        args.prune_epochs = 1
        if args.prediction_network != None:
            prediction_model = torch.load(args.prediction_network, map_location=torch.device(device)).to(device)
            print('predict')
        else:
            raise ValueError("No prediction_network is given")
    
    
    for i in range(args.prune_epochs):
        print_log(f'----------prune epoch{i}----------', logger=logger)
        
        # 定义迭代稀疏的剪枝多少
        sparsity = 1 - (1 - sparse)/(args.prune_epochs - (1 - sparse)*i)
        
        prune_loop()