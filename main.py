
import os
import json

from run_tools import runner_autostrain 
from run_utils.parser import get_args
from run_utils.logger import get_logger
from run_utils.config import load_config
from run_tools import runner_prune


def main():
    args = get_args()
    ## file path
    if args.expid == " ":
        print("WARNING: this experiment is not being saved.")
        setattr(args, 'save', False)
    else:
        result_dir = '{}/{}/{}'.format(args.result_dir, args.experiment, args.expid)   #result_dir 在哪里设定的
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
        
    ## config
    cfgs = load_config(args)
                
    ## save args
    if args.save:
        with open(args.result_dir + '/args.json', 'w') as f:
            json.dump(args.__dict__, f, sort_keys=True, indent=4)
            
    ## Run Experiment
    if args.experiment == 'prune':
        print("Pruning")
        runner_prune.run(cfgs)
    
    elif args.experiment == 'pretrain':
        print("Pretraining")
        runner_autostrain.run(cfgs)
    
    elif args.experiment == 'dataset':
        print("Generating Dataset")
        runner_prune.run(cfgs)
        
    else:
        raise ValueError("Invalid process")


if __name__ == '__main__':
    main()
    
    
    