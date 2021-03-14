import os

import numpy as np

import routing


def test_load_graph(dataset):
    G = routing.load_network_topology(dataset)

    if not os.path.isfile('../../data/topo/{}_segments.npy'.format(dataset)):

        segments = routing.get_segments(G)
        np.save('../../data/topo/{}_segments'.format(dataset), segments)
    else:
        segments = np.load('../../data/topo/{}_segments.npy'.format(dataset), allow_pickle=True)

    print(segments)
    tm = np.ones((12, 12)) * 100
    gt_tms = np.zeros((10, 12, 12))

    solver = routing.HeuristicSolver(G, time_limit=1)
    p_solution = None
    # _s = time.time()
    try:
        solution = solver.solve(tm, solution=p_solution)  # solve backtrack solution (line 131)
    except:
        solution = solver.initialize()


if __name__ == '__main__':
    test_load_graph(dataset='abilene_tm')
