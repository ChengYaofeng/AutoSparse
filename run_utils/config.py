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
        cfgs['train']['train_batchsize'] = args.batchsize
        
    if args.expid is not None:
        cfgs['policy']['expid'] = args.expid
        
    if args.autos_model is not None:
        cfgs['policy']['autos_model'] = args.autos_model
    
    if args.compression is not None:        
        cfgs['prune']['compression'] = args.compression
        
    if args.save_important is not None:
        cfgs['policy']['save_important'] = args.save_important
    
    if args.seed is not None:
        cfgs['policy']['seed'] = args.seed
    
    if args.schedule is not None:
        cfgs['prune']['schedule'] = args.schedule
    
    if args.save_important is not None:
        cfgs['policy']['save_important'] = args.save_important
        
    return cfgs