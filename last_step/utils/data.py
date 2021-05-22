import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class MinMaxScaler_torch():

    def __init__(self, min=None, max=None, device='cuda:0'):
        self.min = min
        self.max = max
        self.device = device

    def fit(self, data):
        self.min = torch.min(data)
        self.max = torch.max(data)

    def transform(self, data):
        _data = data.clone()
        return (_data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return (data * (self.max - self.min + 1e-8)) + self.min


class StandardScaler_torch():

    def __init__(self):
        self.means = 0
        self.stds = 0

    def fit(self, data):
        self.means = torch.mean(data, dim=0)
        self.stds = torch.std(data, dim=0)

    def transform(self, data):
        _data = data.clone()
        data_size = data.size()

        if len(data_size) > 2:
            _data = _data.reshape(-1, data_size[-1])

        _data = (_data - self.means) / (self.stds + 1e-8)

        if len(data_size) > 2:
            _data = _data.reshape(data.size())

        return _data

    def inverse_transform(self, data):
        data_size = data.size()
        if len(data_size) > 2:
            data = data.reshape(-1, data_size[-1])

        data = (data * (self.stds + 1e-8)) + self.means

        if len(data_size) > 2:
            data = data.reshape(data_size)

        return data


def granularity(data, k):
    if k == 1:
        return data
    else:
        newdata = [np.mean(data[i:i + k], axis=0) for i in range(0, data.shape[0], k)]
        newdata = np.asarray(newdata)
        print('new data: ', newdata.shape)
        return newdata


class TrafficDataset(Dataset):

    def __init__(self, X, args):
        # save parameters
        self.args = args

        self.out_seq_len = args.out_seq_len
        self.k = args.k  # granularity

        self.oX = np.copy(X)
        self.oX = self.np2torch(self.oX)

        # generate data with granularity
        self.X = granularity(X, self.k)
        self.X = self.np2torch(self.X)

        self.n_timeslots, self.n_series = self.X.shape

        # get valid start indices for sub-series
        self.indices = self.get_indices()

        if torch.isnan(self.X).any():
            raise ValueError('Data has Nan')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        xgt = self.oX[t * self.k:(t + self.args.seq_len_x) * self.k]  # step: t-> t + seq_x
        # ground truth data for doing traffic engineering
        y_gt = self.oX[(t + self.args.seq_len_x) * self.k:
                       (t + self.args.seq_len_x + self.args.seq_len_y) * self.k]

        sample = {'x_gt': xgt, 'y_gt': y_gt}
        return sample

    def np2torch(self, X):
        X = torch.Tensor(X)
        if torch.cuda.is_available():
            X = X.to(self.args.device)
        return X

    def get_indices(self):
        T, D = self.X.shape
        indices = np.arange(T - self.args.seq_len_x - self.args.seq_len_y)
        return indices


def load_matlab_matrix(path, variable_name):
    X = loadmat(path)[variable_name]
    return X


def load_raw(args):
    # load ground truth
    path = args.datapath

    data_path = os.path.join(path, 'data/{}.mat'.format(args.dataset))
    X = load_matlab_matrix(data_path, 'X')
    if len(X.shape) > 2:
        X = np.reshape(X, newshape=(X.shape[0], -1))

    return X


def train_test_split(X):
    train_size = int(X.shape[0] * 0.5)
    val_size = int(X.shape[0] * 0.1)
    test_size = X.shape[0] - train_size - val_size

    if train_size >= 7000:
        train_size = 7000
    if val_size >= 1000:
        val_size = 1000

    if test_size >= 2000:
        test_size = 2000

    X_train = X[:train_size]

    X_val = X[train_size:val_size + train_size]

    X_test_list = []
    for i in range(5):
        X_test = X[val_size + train_size + test_size * i: val_size + train_size + test_size * (i + 1)]
        X_test_list.append(X_test)
        if val_size + train_size + test_size * (i + 1) >= X.shape[0]:
            break

    return X_train, X_val, X_test_list


def get_dataloader(args):
    # loading data
    X = load_raw(args)

    train, val, test_list = train_test_split(X)

    # Training set
    train_set = TrafficDataset(train, args=args)
    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch_size,
                              shuffle=True)

    # validation set
    val_set = TrafficDataset(val, args=args)
    val_loader = DataLoader(val_set,
                            batch_size=args.val_batch_size,
                            shuffle=False)

    test_set = TrafficDataset(test_list[args.testset], args=args)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader
