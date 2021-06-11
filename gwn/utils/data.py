import os
import pickle

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class MinMaxScaler_torch:

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


class StandardScaler_torch:

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
        return np.copy(data)
    else:
        newdata = [np.mean(data[i:i + k], axis=0) for i in range(0, data.shape[0], k)]
        newdata = np.asarray(newdata)
        print('new data: ', newdata.shape)
        return newdata


class TrafficDataset(Dataset):

    def __init__(self, dataset, args):
        # save parameters
        self.args = args

        self.type = args.type
        self.out_seq_len = args.out_seq_len
        self.X = self.np2torch(dataset['X'])
        self.Y = self.np2torch(dataset['Y'])
        self.Xgt = self.np2torch(dataset['Xgt'])
        self.Ygt = self.np2torch(dataset['Ygt'])
        self.scaler = dataset['Scaler']

        self.nsample, self.len_x, self.nflows, self.nfeatures = self.X.shape

        # get valid start indices for sub-series
        self.indices = self.get_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x = self.X[t]
        y = self.Y[t]
        xgt = self.Xgt[t]
        ygt = self.Ygt[t]
        sample = {'x': x, 'y': y, 'x_gt': xgt, 'y_gt': ygt}
        return sample

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def np2torch(self, X):
        X = torch.Tensor(X)
        if torch.cuda.is_available():
            X = X.to(self.args.device)
        return X

    def get_indices(self):
        indices = np.arange(self.nsample)
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


def np2torch(X, device):
    X = torch.Tensor(X)
    if torch.cuda.is_available():
        X = X.to(device)
    return X


def data_preprocessing(data, args, gen_times=5, scaler=None):
    n_timesteps, n_series = data.shape

    # original dataset with granularity k = 1
    oX = np.copy(data)
    oX = np2torch(oX, args.device)

    # Obtain data with different granularity k
    X = granularity(data, args.k)
    X = np2torch(X, args.device)

    # scaling data
    if scaler is None:
        scaler = StandardScaler_torch()
        scaler.fit(X)
    else:
        scaler = scaler

    X_scaled = scaler.transform(X)

    # if args.tod:
    #     tod = get_tod(n_timesteps, n_series, args.day_size, args.device)
    #
    # if args.ma:
    #     ma = get_ma(X, args.seq_len_x, n_timesteps, args.device)
    #
    # if args.mx:
    #     mx = get_mx(X, args.seq_len_x, n_timesteps, args.device)
    #
    # if np.isnan(X).any():
    #     raise ValueError('Data has Nan')

    len_x = args.seq_len_x
    len_y = args.seq_len_y

    dataset = {'X': [], 'Y': [], 'Xgt': [], 'Ygt': [], 'Scaler': scaler}

    skip = 4
    start_idx = 0
    for _ in range(gen_times):
        for t in range(start_idx, n_timesteps - len_x - len_y, len_x):
            print(t)
            x = X_scaled[t:t + len_x]
            x = x.unsqueeze(dim=-1)  # add feature dim [seq_x, n, 1]

            y = torch.max(X[t + len_x:t + len_x + len_y], dim=0)[0]
            y = y.reshape(1, -1)

            # Data for doing traffic engineering
            x_gt = oX[t * args.k:(t + len_x) * args.k]
            y_gt = oX[(t + len_x) * args.k: (t + len_x + len_y) * args.k]

            dataset['X'].append(x)  # [sample, len_x, k, 1]
            dataset['Y'].append(y)  # [sample, 1, k]
            dataset['Xgt'].append(x_gt)
            dataset['Ygt'].append(y_gt)

        start_idx = start_idx + skip

    dataset['X'] = torch.stack(dataset['X'], dim=0)
    dataset['Y'] = torch.stack(dataset['Y'], dim=0)
    dataset['Xgt'] = torch.stack(dataset['Xgt'], dim=0)
    dataset['Ygt'] = torch.stack(dataset['Ygt'], dim=0)

    dataset['X'] = dataset['X'].cpu().data.numpy()
    dataset['Y'] = dataset['Y'].cpu().data.numpy()
    dataset['Xgt'] = dataset['Xgt'].cpu().data.numpy()
    dataset['Ygt'] = dataset['Ygt'].cpu().data.numpy()

    print('   X: ', dataset['X'].shape)
    print('   Y: ', dataset['Y'].shape)
    print('   Xgt: ', dataset['Xgt'].shape)
    print('   Ygt: ', dataset['Ygt'].shape)

    return dataset


def train_test_split(X):
    train_size = int(X.shape[0] * 0.5)
    val_size = int(X.shape[0] * 0.1)
    test_size = X.shape[0] - train_size - val_size

    if train_size >= 7000:
        train_size = 7000
    if val_size >= 1000:
        val_size = 1000

    if test_size >= 1000:
        test_size = 1000

    X_train = X[:train_size]

    X_val = X[train_size:val_size + train_size]

    X_test_list = []
    for i in range(10):
        X_test = X[val_size + train_size + test_size * i: val_size + train_size + test_size * (i + 1)]
        X_test_list.append(X_test)
        if val_size + train_size + test_size * (i + 1) >= X.shape[0]:
            break

    return X_train, X_val, X_test_list


def get_dataloader(args):
    # loading data
    X = load_raw(args)
    total_timesteps, total_series = X.shape
    stored_path = os.path.join(args.datapath, 'data/preprocessed_{}_{}_{}/'.format(args.dataset, args.seq_len_x,
                                                                                   args.seq_len_y))
    if not os.path.exists(stored_path):
        os.makedirs(stored_path)

    saved_train_path = os.path.join(stored_path, 'train.pkl')
    saved_val_path = os.path.join(stored_path, 'val.pkl')
    saved_test_path = os.path.join(stored_path, 'test.pkl')

    if not os.path.exists(saved_train_path):
        train, val, test_list = train_test_split(X)
        trainset = data_preprocessing(data=train, args=args, gen_times=5, scaler=None)
        train_scaler = trainset['Scaler']
        with open(saved_train_path, 'wb') as fp:
            pickle.dump(trainset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

        print('Data preprocessing: VALSET')
        valset = data_preprocessing(data=val, args=args, gen_times=5, scaler=train_scaler)
        with open(saved_val_path, 'wb') as fp:
            pickle.dump(valset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

        testset_list = []
        for i in range(len(test_list)):
            print('Data preprocessing: TESTSET {}'.format(i))
            testset = data_preprocessing(data=test_list[i], args=args, gen_times=1, scaler=train_scaler)
            testset_list.append(testset)

        with open(saved_test_path, 'wb') as fp:
            pickle.dump(testset_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()
    else:
        print('Load saved dataset from {}'.format(stored_path))
        with open(saved_train_path, 'rb') as fp:
            trainset = pickle.load(fp)
            fp.close()
        with open(saved_val_path, 'rb') as fp:
            valset = pickle.load(fp)
            fp.close()
        with open(saved_test_path, 'rb') as fp:
            testset_list = pickle.load(fp)
            fp.close()

    # Training set
    train_set = TrafficDataset(trainset, args=args)
    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch_size,
                              shuffle=True)

    # validation set
    val_set = TrafficDataset(valset, args=args)
    val_loader = DataLoader(val_set,
                            batch_size=args.val_batch_size,
                            shuffle=False)

    test_set = TrafficDataset(testset_list[args.testset], args=args)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader, total_timesteps, total_series
