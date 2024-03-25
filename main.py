
import os
import json

from run_utils.parser import get_args
from run_utils.logger import get_logger
from run_tools import singleshot




if __name__ == '__main__':
    
    args = get_args()
    ## 
    if args.expid == " ":
        print("WARNING: this experiment is not being saved.")
        setattr(args, 'save', False)
    else:
        result_dir = '{}/{}/{}'.format(args.result_dir, args.experiment, args.expid)   #result_dir 在哪里设定的
        setattr(args, 'save', True)
        setattr(args, 'result_dir', result_dir)
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            val = ""
            while val not in ['yes', 'no', 'y', 'n']:
                val = input("Experiment '{}' with expid '{}' exists.  Overwrite (yes/no)? ".format(args.experiment, args.expid))
            if val == 'no' or val == 'n':
                quit()
                
    ## Save Args
    if args.save:
        with open(args.result_dir + '/args.json', 'w') as f:
            json.dump(args.__dict__, f, sort_keys=True, indent=4)
            
    ## Run Experiment
    if args.experiment == 'singleshot':
        for i in range(10):                       #这里循环的意思是
            singleshot.run(args, i)
            
    # if args.experiment == 'multishot':
    #     multishot.run(args)
        
    # if args.experiment == 'unit-conservation':
    # 	unit_conservation.run(args)
     
    # if args.experiment == 'layer-conservation':
    #     layer_conservation.run(args)
        
    # if args.experiment == 'imp-conservation':
    #     imp_conservation.run(args)
        
    # if args.experiment == 'schedule-conservation':
    #     schedule_conservation.run(args)
    