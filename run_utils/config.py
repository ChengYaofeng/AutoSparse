import os
import yaml

def load_config(args):
    '''
    params from config file
    '''
    
    # load config file
    with open(os.path.join(os.getcwd(), args.cfg), 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.SafeLoader)
    
    # update config with args
    cfgs['policy']['result_dir'] = args.result_dir
    
    if args.lr is not None:
        cfgs['train']['lr'] = args.lr
    
    if args.batchsize is not None:
        cfgs['train']['train_batch_size'] = args.batchsize
    
    return cfgs