
import os
import json
import yaml

from run_tools import runner_autostrain 
from run_utils.parser import get_args
from run_utils.logger import get_logger
from run_utils.config import load_config
from run_tools import runner_prune


def main():
    
    args = get_args()
    ## config
    cfgs = load_config(args)
    
    ## Run Experiment
    ## file path
    if args.expid == " ":
        print("WARNING: this experiment is not being saved.")
        setattr(args, 'save', False)
    else:
        
        if args.experiment == 'prune':
            check_dir('prune_results', args, cfgs)
            print('-'*20, "Pruning", '-'*20)
            for i in range(len(cfgs['prune']['compression'])):
                # print(f'times{i}')
                cfgs['policy']['times'] = i
                print('-'*10, 'Compression Level {}'.format(cfgs['prune']['compression'][i]), '-'*10)
                runner_prune.run(cfgs)
        
        elif args.experiment == 'pretrain':
            check_dir('pretrain_results', args, cfgs)
            print('-'*20, 'Pretraining', '-'*20)
            runner_autostrain.run(cfgs)
    
        elif args.experiment == 'dataset':
            check_dir('dataset_results', args, cfgs)
            print('-'*20, "Generating Dataset", '-'*20)
            runner_prune.run(cfgs)
            
        else:
            raise ValueError("Invalid process")
        

def check_dir(exp, args, cfgs):
    major_dir = f'experiment/{exp}/{cfgs["train"]["dataset"]}_{cfgs["train"]["model"]}_{cfgs["prune"]["pruner"]}'
    try:
        os.makedirs(major_dir)
    except FileExistsError:
        pass
    
    result_dir = f"{major_dir}/batch{cfgs['train']['train_batchsize']}_lr{cfgs['train']['lr']}_seed{cfgs['policy']['seed']}_{cfgs['policy']['expid']}"
                   #result_dir 在哪里设定的
    setattr(args, 'save', True)
    setattr(args, 'result_dir', result_dir) 
        ## 调试结束后记得保留回来
    try:
        os.makedirs(result_dir)
    except FileExistsError:
        val = ""
        while val not in ['yes', 'no', 'y', 'n']:
            val = input("Experiment '{}' with expid '{}' exists.  Overwrite (yes(y)/no(n))? ".format(args.experiment, args.expid))
        if val == 'no' or val == 'n':
            quit()
    
    ##
    cfgs['policy']['result_dir'] = args.result_dir
    cfgs['policy']['save'] = args.save
    
    ## save args
    if args.save:
        with open(os.path.join(args.result_dir, os.path.basename(args.cfg)), 'w') as f:
            yaml.dump(cfgs, f)
    

if __name__ == '__main__':
    main()
    
    
    