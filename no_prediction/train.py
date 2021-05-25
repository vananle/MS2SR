import sys

sys.path.append('..')

import time
import torch
import utils
from routing import *

import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def main(args, **model_kwargs):
    device = torch.device(args.device)
    args.device = device
    if 'abilene' in args.dataset:
        args.nNodes = 12
        args.day_size = 288
    elif 'geant' in args.dataset:
        args.nNodes = 22
        args.day_size = 96
    elif 'brain' in args.dataset:
        args.nNodes = 9
        args.day_size = 1440
    elif 'sinet' in args.dataset:
        args.nNodes = 74
        args.day_size = 288
    elif 'renater' in args.dataset:
        args.nNodes = 30
        args.day_size = 288
    elif 'surfnet' in args.dataset:
        args.nNodes = 50
        args.day_size = 288
    elif 'uninett' in args.dataset:
        args.nNodes = 74
        args.day_size = 288
    else:
        raise ValueError('Dataset not found!')

    train_loader, val_loader, test_loader = utils.get_dataloader(args)

    args.train_size, args.nSeries = train_loader.dataset.X.shape
    args.val_size = val_loader.dataset.X.shape[0]
    args.test_size = test_loader.dataset.X.shape[0]
    te_step = args.test_size if args.te_step = 0 else args.te_step
    args.te_step = te_step

    in_dim = 1
    if args.tod:
        in_dim += 1
    if args.ma:
        in_dim += 1
    if args.mx:
        in_dim += 1

    args.in_dim = in_dim

    logger = utils.Logger(args)

    utils.print_args(args)

    x_gt = []
    y_gt = []
    for _, batch in enumerate(test_loader):
        x_gt.append(batch['x_gt'])
        y_gt.append(batch['y_gt'])

    x_gt = torch.cat(x_gt, dim=0)
    y_gt = torch.cat(y_gt, dim=0)

    # run TE
    if args.run_te != 'None':
        x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
        y_gt = y_gt.cpu().data.numpy()
        run_te(x_gt, y_gt, None, args)


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} mins".format(mins))
