import argparse
import os

import numpy as np

import routing_undirected as routing


def test_load_graph(dataset, datapath, args):
    G = routing.load_network_topology(dataset, datapath)

    if not os.path.exists(os.path.join(datapath, 'topo/segments')):
        os.makedirs(os.path.join(datapath, 'topo/segments'))

    segments = routing.get_segments(G)

    print(G.edges)

    # path = os.path.join(datapath, 'topo/segments/{}_digraph'.format(dataset))
    # if not os.path.isfile(path + '.npy'):
    #     segments = routing.get_segments(G)
    #     np.save(path, segments)
    # else:
    #     segments = np.load(path + '.npy', allow_pickle=True)

    print(segments)
    tm = np.ones((12, 12)) * 100
    gt_tms = np.zeros((10, 12, 12))

    solver = routing.MultiStepSRSolver(G, segments)
    p_solution = None
    # _s = time.time()
    solution = solver.solve(tm)  # solve backtrack solution (line 131)


def test_ls2sr(args):
    G = routing.load_network_topology(args.dataset, args.datapath)
    print(G.edges)

    solver = routing.LS2SRSolver(graph=G, args=args)
    p_solution = None
    tm = np.ones((12, 12)) * 100
    solution = solver.solve(tm, p_solution)  # solve backtrack solution (line 131)


def get_args():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='abilene_tm',
                        choices=['abilene_tm', 'geant_tm', 'brain_tm', 'renater_tm', 'surfnet_tm', 'uninett_tm'],
                        help='Dataset, (default abilene_tm)')
    parser.add_argument('--datapath', type=str, default='../../data')
    parser.add_argument('--timeout', type=float, default=10.0)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # test_load_graph(dataset='abilene_tm', datapath='../../data/')
    args = get_args()
    test_ls2sr(args)
