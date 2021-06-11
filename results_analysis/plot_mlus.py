import os

import numpy as np
from matplotlib import pyplot as plt

dataset = 'abilene'

len_x = 12
len_y = 12
k = 1

datapath = 'results/gwn_{}_tm.{}_{}_{}_mae_p2'.format(dataset, k, len_x, len_y)

testsets = [0, 1, 2]

# types = ['p0_optimal', 'p1', 'p2', 'laststep', 'ls2sr_last_step', 'gwn_ls2sr']
types = ['ls2sr_last_step', 'gwn_ls2sr']

for testset in testsets:
    mlus = {}

    for type in types:
        mlu = np.load(os.path.join(datapath, 'test_{}_{}_mlus.npy'.format(testset, type)))
        mlus[type] = mlu.flatten()

    ntimesteps = mlus[types[0]].shape[0]
    T = 100
    for t in range(0, ntimesteps, T):
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 4), tight_layout=True)
        for type in types:
            plt.plot(mlus[type][t:t + T], label=type)

        plt.xlabel('Timestep')
        plt.ylabel('Utilization')
        plt.title('Dataset {} - Testset {} - {} - {}'.format(dataset, testset, t, t + T))
        plt.legend()
        plt.savefig('figures/mlu-{}-{}-{}-{}.png'.format(dataset, testset, t, t + T))
        # plt.savefig('figures/mlu-{}-{}-{}-{}.eps'.format(dataset, testset, t, t+T))
        plt.cla()
