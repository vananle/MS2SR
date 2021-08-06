import sys

sys.path.append('..')

import time
import models
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

    args.do_graph_conv = True
    args.aptonly = True
    args.addaptadj = True
    args.randomadj = True
    args.train_batch_size = 64
    args.val_batch_size = 64
    args.dataset = 'abilene_tm'
    args.device = 'cuda:0'

    sets = ['train', 'val', 'test_0', 'test_1', 'test_2', 'test_3', 'test_4']
    y_gt_train = []
    for set in sets:
        if set == 'train' or set == 'val':
            args.test = False
            args.testset = 0
        else:
            args.test = True
            testset = int(set.split('_')[1])
            args.testset = testset

        train_loader, val_loader, test_loader, total_timesteps, total_series = utils.get_dataloader(args)

        args.train_size, args.nSeries = train_loader.dataset.nsample, train_loader.dataset.nflows
        args.val_size = val_loader.dataset.nsample
        args.test_size = test_loader.dataset.nsample
        args.te_step = args.test_size

        in_dim = 1
        if args.tod:
            in_dim += 1
        if args.ma:
            in_dim += 1
        if args.mx:
            in_dim += 1

        args.in_dim = in_dim

        aptinit, supports = utils.make_graph_inputs(args, device)

        model = models.GWNet.from_args(args, supports, aptinit, **model_kwargs)
        model.to(device)
        logger = utils.Logger(args)

        engine = utils.Trainer.from_args(model, train_loader.dataset.scaler, args)

        utils.print_args(args)

        # Metrics on test data

        if set == 'train':
            data_loader = train_loader
        elif set == 'val':
            data_loader = val_loader
        else:
            data_loader = test_loader

        engine.model.load_state_dict(torch.load(logger.best_model_save_path))
        with torch.no_grad():
            test_met_df, x_gt, y_gt, y_real, yhat = engine.test(data_loader, engine.model, args.out_seq_len)

        x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
        y_gt = y_gt.cpu().data.numpy()
        yhat = yhat.cpu().data.numpy()

        # run TE
        if args.run_te != 'None':
            print(' SET: {}'.format(set))
            args.testset = set
            # run_te(x_gt, y_gt, yhat, args)

            te_step = x_gt.shape[0]
            if set == 'train':
                y_gt_train = y_gt
            all_data = np.reshape(y_gt, newshape=(-1, y_gt_train.shape[-1]))
            max_tm = np.max(all_data, axis=0, keepdims=True)

            graphs = createGraph_srls(os.path.join(args.datapath, 'topo/{}_node.csv'.format(args.dataset)),
                                      os.path.join(args.datapath, 'topo/{}_edge.csv'.format(args.dataset)))
            srls_fix_max(max_tm, y_gt, graphs, te_step, args)

if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
