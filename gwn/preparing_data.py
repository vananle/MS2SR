from scipy.io import loadmat, savemat

dataset = 'brain_tm'

data = loadmat('../../data/data/{}.mat'.format(dataset))['X']

data5 = data[0:data.shape[0]:5]
data15 = data[0:data.shape[0]:15]

print(data5.shape)
print(data15.shape)
savemat('../../data/data/brain5_tm.mat', {'X': data5})
