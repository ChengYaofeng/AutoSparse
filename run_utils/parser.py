import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='pretrain',
                        help='experiment name (default: example)')
    parser.add_argument('--expid', type=str, default='',
                        help='name used to save results (default: "")')
    parser.add_argument('--result-dir', type=str, default='experiment/pretrain_resutls',
                        help='path to directory to save results (default: "experiment/pretrain_resutls")')
    parser.add_argument('--gpu', type=int, default='0',
                        help='number of GPU device to use (default: 0)')
    parser.add_argument('--workers', type=int, default='4',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--cfg', type=str, default=None,
                        help='config file path')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batchsize', type=int, default=None,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--autos_model', type=str, default=None,
                        help='autos model path')
    parser.add_argument('--compression', type=float, nargs='*', default=None,
                        help='list of number of prune-train cycles (levels) for multishot (default: [])')
    parser.add_argument('--save_important', type=str, default=None,
                        help='save important in one or not')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--schedule', type=str, default=None, choices=['pct','num'],
                        help='sparse training schedule (default: percent), choices: percent and number')
    args = parser.parse_args()
    
    return args