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

    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()
    dataset_name = args.dataset
    run_te = ['gwn_ls2sr', 'gt_ls2sr', 'p0', 'p1', 'p2', 'gwn_p2', 'p3', 'onestep',
              'laststep', 'laststep_ls2sr', 'firststep', 'or']
    # run_te = ['None', 'gwn_ls2sr', 'gt_ls2sr', 'p0', 'p1', 'p2', 'gwn_p2', 'p3', 'onestep',
    #           'prophet', 'laststep', 'laststep_ls2sr', 'firststep', 'or']
    if args.test:
        testset = [0, 1, 2, 3, 4, 5]
    else:
        testset = [0]

    iteration = trange(len(testset))
    # experiment for each dataset
    for test in iteration:
        cmd = 'python train.py --do_graph_conv --aptonly --addaptadj --randomadj'
        cmd += ' --train_batch_size 128 --val_batch_size 128'
        cmd += ' --dataset {}'.format(dataset_name)
        cmd += ' --device {}'.format(args.device)
        if args.test:
            cmd += ' --test'
            cmd += ' --testset {}'.format(test)
            for te in run_te:
                cmd += ' --run_te {}'.format(te)
                os.system(cmd)
                iteration.set_description(
                    'Dataset {} - testset {} te: {}'.format(dataset_name, test.te))

        else:
            os.system(cmd)
            iteration.set_description(
                'Dataset {} - testset {}'.format(dataset_name, test))


if __name__ == '__main__':
    main()
