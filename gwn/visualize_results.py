import sys
import time

sys.path.append('..')

import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import utils


def main(args, **model_kwargs):
    device = torch.device(args.device)
    args.device = device
    if args.dataset == 'abilene_tm':
        args.nNodes = 12
        args.day_size = 288
    elif args.dataset == 'geant_tm':
        args.nNodes = 22
        args.day_size = 96
    elif args.dataset == 'brain_tm':
        args.nNodes = 9
        args.day_size = 1440
    else:
        raise ValueError('Dataset not found!')

    in_dim = 1
    if args.tod:
        in_dim += 1
    if args.ma:
        in_dim += 1
    if args.mx:
        in_dim += 1

    args.in_dim = in_dim

    logger = utils.Logger(args)


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
