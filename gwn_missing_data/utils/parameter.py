import argparse

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']


def get_args():
    # create argument parser
    parser = argparse.ArgumentParser()

    # parameter for dataset
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='abilene_tm',
                        choices=['abilene_tm', 'geant_tm', 'brain_tm'],
                        help='Dataset, (default abilene_tm)')
    parser.add_argument('--sr', type=float, default=0.4, help='Sampling rate (1.0 - missing_rate)')

    parser.add_argument('--datapath', type=str, default='../../data')
    parser.add_argument('--type', type=str, default='p2', choices=['p1', 'p2', 'p3'],
                        help='problem formulation (default p2)')
    parser.add_argument('--miss_type', type=str, default='uniform', choices=['uniform', 'block'],
                        help='problem formulation (default p2)')
    parser.add_argument('--trunk', type=int, default=3, help='trunk for p3 problem (default 3)')
    parser.add_argument('--k', type=int, default=1, help='granularity scale', choices=[1, 2, 3])

    parser.add_argument('--tod', action='store_true')
    parser.add_argument('--ma', action='store_true')
    parser.add_argument('--mx', action='store_true')

    # Model
    parser.add_argument('--model', type=str, default='gwn', choices=['gwn', 'gcrint'])
    parser.add_argument('--lstm_hidden', type=int, default=32)
    parser.add_argument('--seq_len_x', type=int, default=64, help='input length default 64')
    parser.add_argument('--seq_len_y', type=int, default=12, help='routing cycle 12')

    # Gcrint
    parser.add_argument('--n_lstm', type=int, default=2)

    # Graph
    parser.add_argument('--adjdata', type=str, default='../../data/data/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type (default doubletransition)',
                        choices=ADJ_CHOICES)
    parser.add_argument('--apt_size', default=10, type=int, help='default 10')

    # Wavenet

    parser.add_argument('--dilation_channels', type=int, default=32, help='inputs dimension (default 32)')
    parser.add_argument('--residual_channels', type=int, default=32, help='inputs dimension')
    parser.add_argument('--skip_channels', type=int, default=256, help='inputs dimension')
    parser.add_argument('--end_channels', type=int, default=512, help='inputs dimension')

    parser.add_argument('--blocks', type=int, default=5, help='')
    parser.add_argument('--layers', type=int, default=2, help='')
    parser.add_argument('--hidden', type=int, default=32, help='Number of channels for internal conv')
    parser.add_argument('--kernel_size', type=int, default=4, help='kernel_size for internal conv')
    parser.add_argument('--stride', type=int, default=4, help='stride for internal conv')
    parser.add_argument('--n_obs', default=None, help='Only use this many observations. For unit testing.')
    parser.add_argument('--cat_feat_gc', action='store_true')

    # loss
    parser.add_argument('--loss_fn', type=str, default='mae', choices=['mse', 'mae', 'mse_u', 'mae_u'])
    parser.add_argument('--lamda', type=float, default=2.0)
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

    # training
    parser.add_argument('--train_bs', type=int, default=256)
    parser.add_argument('--val_bs', type=int, default=256)
    parser.add_argument('--test_bs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
    parser.add_argument('--patience', type=int, default=20, help='quit if no improvement after this many iterations')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot', action='store_true')

    # parameter for test_routing
    parser.add_argument('--run_te', action='store_true')

    # parser.add_argument('--test_routing', type=str, default='sr',
    #                     choices=['sr', 'sp', 'or', 'ta'])
    # parser.add_argument('--mon_policy', type=str, default='random',
    #                     choices=['heavy_hitter', 'fluctuation', 'fgg', 'random'])
    parser.add_argument('--te_step', type=int, default=0)

    # get args
    args = parser.parse_args()

    if args.seq_len_x <= 45:
        args.blocks = 3

    if args.type == 'p1':
        args.out_seq_len = args.seq_len_y
    elif args.type == 'p2':
        args.out_seq_len = 1
    elif args.type == 'p3':
        if args.seq_len_y % args.trunk != 0:
            args.seq_len_y = int(args.seq_len_y / args.trunk) * args.trunk
        args.out_seq_len = args.trunk

    return args


def print_args(args):
    print('-------------------------------------')
    print('[+] Time-series recovering experiment')
    if args.test:
        print('|--- Run Test')
    else:
        print('|--- Run Train')

    print('---------------------------------------------------------')
    print('[+] Time-series prediction experiment')
    print('---------------------------------------------------------')
    print('    - dataset                :', args.dataset)
    print('    - granularity scale      :', args.k)
    print('    - num_series             :', args.nSeries)
    print('    - train size             : {}x{}'.format(args.train_size, args.nSeries))
    print('    - val size               : {}x{}'.format(args.val_size, args.nSeries))
    print('    - test size              : {}x{}'.format(args.test_size, args.nSeries))
    print('    - log path               :', args.log_dir)
    print('---------------------------------------------------------')
    print('    - model                  :', args.model)
    print('    - num layers             :', args.n_lstm)
    print('    - wn_blocks              :', args.blocks)
    print('    - wn_layers              :', args.layers)
    print('    - hidden                 :', args.hidden)
    print('    - kernel_size            :', args.kernel_size)
    print('    - stride                 :', args.stride)
    print('    - dilation_channels      :', args.dilation_channels)
    print('    - residual_channels      :', args.residual_channels)
    print('    - end_channels           :', args.end_channels)
    print('    - skip_channels          :', args.skip_channels)
    print('----------------------------')
    print('    - type                   :', args.type)
    print('    - seq_len_x              :', args.seq_len_x)
    print('    - seq_len_y              :', args.seq_len_y)
    print('    - out_seq_len            :', args.out_seq_len)
    print('    - tod                    :', args.tod)
    print('    - ma                     :', args.ma)
    print('    - mx                     :', args.mx)
    print('---------------------------------------------------------')
    print('    - device                 :', args.device)
    print('    - train_bs               :', args.train_bs)
    print('    - val_bs                 :', args.val_bs)
    print('    - test_bs                :', args.test_bs)
    print('    - epochs                 :', args.epochs)
    print('    - learning_rate          :', args.learning_rate)
    print('    - patience               :', args.patience)
    print('    - plot_results           :', args.plot)
    print('---------------------------------------------------------')
    print('    - run te                 :', args.run_te)
    # print('    - routing        :', args.routing)
    # print('    - mon_policy     :', args.mon_policy)
    print('    - te_step                :', args.te_step)
    print('---------------------------------------------------------')
