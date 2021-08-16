import argparse
import os
import subprocess as sp

from tqdm import trange


def call(args):
    p = sp.run(args=args,
               stdout=sp.PIPE,
               stderr=sp.PIPE)
    stdout = p.stdout.decode('utf-8')
    return stdout


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='abilene_tm',
                        choices=['abilene_tm', 'geant_tm', 'brain_tm', 'brain5_tm', 'brain15_tm', 'abilene15_tm',
                                 'brain10_tm', 'abilene10_tm'],
                        help='Dataset, (default abilene_tm)')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--run_te', type=str, choices=['None', 'gwn_ls2sr', 'gt_ls2sr', 'p0', 'p1', 'p2', 'gwn_p2',
                                                       'p3', 'onestep', 'prophet', 'laststep', 'laststep_ls2sr',
                                                       'firststep', 'or', 'gwn_srls', 'gt_srls', 'srls_p0'],
                        default='None')
    parser.add_argument('--testset', type=int, default=-1,
                        choices=[-1, 0, 1, 2, 3, 4],
                        help='Test set, (default -1 run all test)')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--timeout', type=float, default=1.0)
    parser.add_argument('--nrun', type=int, default=3)

    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()
    dataset_name = args.dataset
    if args.run_te == 'None':
        run_te = ['gwn_ls2sr', 'gt_ls2sr', 'p0', 'p1', 'p2', 'gwn_p2', 'p3', 'onestep',
                  'laststep', 'laststep_ls2sr', 'firststep', 'or']
    else:
        run_te = [args.run_te]
    # run_te = ['None', 'gwn_ls2sr', 'gt_ls2sr', 'p0', 'p1', 'p2', 'gwn_p2', 'p3', 'onestep',
    #           'prophet', 'laststep', 'laststep_ls2sr', 'firststep', 'or']
    if args.test:
        if args.testset == -1:
            testset = [0, 1, 2, 3, 4]
        else:
            testset = [args.testset]

    else:
        testset = [0]

    iteration = trange(len(testset))
    # experiment for each dataset
    for test in iteration:
        cmd = 'python train.py --do_graph_conv --aptonly --addaptadj --randomadj'
        cmd += ' --train_batch_size 64 --val_batch_size 64'
        cmd += ' --dataset {}'.format(dataset_name)
        cmd += ' --device {}'.format(args.device)
        cmd += ' --epochs {}'.format(args.epochs)
        if args.test:
            cmd += ' --test'
            cmd += ' --testset {}'.format(testset[test])
            for te in run_te:
                cmd += ' --run_te {}'.format(te)
                cmd += ' --timeout {}'.format(args.timeout)
                cmd += ' --nrun {}'.format(args.nrun)
                print(cmd)
                os.system(cmd)
                iteration.set_description(
                    'Dataset {} - testset {} te: {}'.format(dataset_name, testset[test], te))

        else:
            print(cmd)
            os.system(cmd)
            iteration.set_description(
                'Dataset {} - testset {}'.format(dataset_name, testset[test]))


if __name__ == '__main__':
    main()
