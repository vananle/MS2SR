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

    train_loader, val_loader, test_loader, total_timesteps, total_series = utils.get_dataloader(args)

    args.test_size = test_loader.dataset.nsample
    args.te_step = args.test_size
    args.nSeries = total_series

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

    fnames = ['train', 'val']
    data_loaders = [train_loader, val_loader, test_loader]
    for i in range(2):
        data_loader = data_loaders[i]

        # Metrics on test data
        x_gt = []
        y_gt = []
        for _, batch in enumerate(data_loader):
            x_gt.append(batch['x_gt'])
            y_gt.append(batch['y_gt'])

        x_gt = torch.cat(x_gt, dim=0)
        y_gt = torch.cat(y_gt, dim=0)

        x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
        y_gt = y_gt.cpu().data.numpy()
        np.save(os.path.join(logger.log_dir, 'x_gt_test'), x_gt)
        np.save(os.path.join(logger.log_dir, 'y_gt_test'), y_gt)
        # run TE
        args.run_te = 'vae_gen_data'
        if args.run_te != 'None':
            te_step = x_gt.shape[0]

            graphs = createGraph_srls(os.path.join(args.datapath, 'topo/{}_node.csv'.format(args.dataset)),
                                      os.path.join(args.datapath, 'topo/{}_edge.csv'.format(args.dataset)))
            vae_no_pred_gen_data(x_gt, y_gt, graphs, te_step, args, fnames[i])


from datetime import date

if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())
