import os

import numpy as np

import routing


def test_load_graph(dataset, datapath):
    G = routing.load_network_topology(dataset, datapath)

    if not os.path.exists(os.path.join(datapath, 'topo/segments')):
        os.makedirs(os.path.join(datapath, 'topo/segments'))

    path = os.path.join(datapath, 'topo/segments/{}_digraph'.format(dataset))
    if not os.path.isfile(path + '.npy'):
        segments = routing.get_segments(G)
        np.save(path, segments)
    else:
        segments = np.load(path + '.npy', allow_pickle=True)

    print(segments)
    tm = np.ones((12, 12)) * 100
    gt_tms = np.zeros((10, 12, 12))

    solver = routing.MultiStepSRSolver(G, segments)
    p_solution = None
    # _s = time.time()
    solution = solver.solve(tm)  # solve backtrack solution (line 131)


if __name__ == '__main__':
    test_load_graph(dataset='abilene_tm', datapath='../../data/')
