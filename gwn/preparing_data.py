import numpy as np
from scipy.io import loadmat, savemat

dataset = 'brain_tm'

data = loadmat('../../data/data/{}.mat'.format(dataset))['X']

data5 = [np.mean(data[i:i + 5], axis=0) for i in range(0, data.shape[0], 5)]
data5 = np.asarray(data5)

print('Brain5', data5.shape)
savemat('../../data/data/brain5_tm.mat', {'X': data5})

dataset = 'abilene_tm'

data = loadmat('../../data/data/{}.mat'.format(dataset))['X']

data15 = [np.mean(data[i:i + 3], axis=0) for i in range(0, data.shape[0], 3)]
data15 = np.asarray(data15)
print('Abilene15', data15.shape)
savemat('../../data/data/abilene15_tm.mat', {'X': data15})
