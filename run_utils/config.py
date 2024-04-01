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
        
    
    return cfgs