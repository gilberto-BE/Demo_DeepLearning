# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:16:08 2018

@author: gilbe
"""
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, Imputer, QuantileTransformer
from torch.utils.data import DataLoader


use_cuda = torch.cuda.is_available()

class LSTMLayers(nn.Module):
    
    def __init__(self, input_size=1, hidden_units=10,
                 num_layers=1, batch=2**6, dropout=0.4,
                 num_directions=1, batch_first=False):
        super(LSTMLayers, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.batch = batch
        self.dropout = dropout
        self.num_directions = num_directions
        self.batch_first = batch_first
        self.drop_rate = nn.Dropout(self.dropout)

        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_units,
                            self.num_layers,
                            dropout=self.dropout,
                            batch_first=self.batch_first)
        """ out_featuers = 1 for time series
        Input: (N,∗,in_features) where * means any number
        of additional dimensions

        Output: (N,∗,out_features) where all but the last
        dimension are the same shape as """
        self.linear = nn.Linear(in_features=self.hidden_units,
                                out_features=1)

        if use_cuda:
            """ there are 2 elements in self.hidden"""
            self.lstm = self.lstm.cuda()
            self.linear = self.linear.cuda()

    def init_hidden(self):
        """
        this code initialises c0 and h0
        return a tuple with 2 tensors/variables"""
        zeros = Variable(
                torch.zeros(self.num_layers * self.num_directions,
                            self.batch,
                            self.hidden_units),
                            requires_grad=False)
        return (zeros, zeros)

    def forward(self, X):
        if use_cuda:
               X = X.cuda()
        out, hidden = self.lstm(X, self.hidden)
        out = self.drop_rate(out)
        out = self.linear(out[-1].view(-1, self.hidden_units))
        return out
#-----------------------------------------------------------------------------------

def to_tensor(data, timesteps=30):
    x = np.array([data[i:i + timesteps]
                  for i in range(len(data) - timesteps)], dtype=float)
    return x


# Get train and development sets
# Note this function assumes a time series, indexing has to be 
# modified if using a dataframe
# x is either price series or return series
def get_train_dev(x, dev=0.85):
#    if to_return:
#        x = to_return(x)
    dev_ix = int(dev * len(x))
    train = x.iloc[:dev_ix]
    dev = x.iloc[dev_ix:]
    
    # impute data
    # dont use pipeline in order to inverse transform
    # reshape([-1, 1]) is only needed in the case of working with a 1-Dim series
    imputer = Imputer(strategy='median')
    train = imputer.fit_transform(train.values.reshape([-1, 1]))
    dev = imputer.transform(dev.values.reshape([-1, 1]))
    
    # normalize
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train = scaler.fit_transform(train.reshape(-1, 1))
    dev = scaler.transform(dev.reshape(-1, 1))
    
    # transform data to tensors
    # for now we index in 2-Dim because we started with
    # 1-Dim time series
    # in case of using a dataframe indexes are 3-Dim
    train = to_tensor(train)
    x_train = train[:,:-1]
    y_train = train[:,-1]
    
    dev = to_tensor(dev)
    x_dev = dev[:, :-1]
    y_dev = dev[:, -1]
    
    return x_train, y_train, x_dev, y_dev


def mini_batch(X, Y, mini_batch_size=64, shuffle=False):
    """ Creates a list of minibatches from (X, Y)
    This function a new version for the keras model where data
    has to be structure differently
    Arguments:
    X -- input data of shape (number of examples, input_dim)
    Y -- true label vector of shape (number o examples, 1)
    mini_batch_size -- size of mini-batches, integer
    Return:
    A list with minibatches for X and Y
    """
    # New code to handle many columns: 2018-03-22
    Y = Y.reshape(Y.shape[0], -1)

    print('type(X):', type(X))
    print('dim(X):', X.shape)
    m = len(X) # this is the total number of examples
    mini_batches = []
    # TODO: Complete the if case
    if shuffle:
        seed = 0
        np.random.seed(seed)

    # number of complete
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch_Y = Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = X[:m - mini_batch_size * math.floor(m/mini_batch_size), :]
        mini_batch_Y = Y[:m - mini_batch_size * math.floor(m/mini_batch_size), :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def to_return(x, period=1):
    """ This function supposes that the input is a
    dataframe or series"""
    return (x - x.shift(periods=period, axis='index'))/x.shift(periods=period)


if __name__ == '__main__':
    data = pd.read_csv('closing_prices_tiingo.csv', 
                       parse_dates=True, 
                       infer_datetime_format=True)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.head()
    
    aapl = data['AAPL']
    ret = to_return(aapl)
    x_train, y_train, x_dev, y_dev = get_train_dev(ret)
    
    print('type of x_train', type(x_train[0]))
    print(x_train[0])
    print(x_train.shape)
    print(y_train.shape)
    print(x_dev.shape)
    print(y_dev.shape)
    print('')
    print('Check the minibatch:')
    
    
    MINIBATCH = 2**4
    mini_batches_train = mini_batch(x_train, y_train, mini_batch_size=MINIBATCH)
    print('type mini-batch:', type(mini_batches_train))
    print('x[0]-element:')
    print(type(mini_batches_train[0][0]))
    print(mini_batches_train[0][0].shape)
    #print(mini_batches_train[0][0])
    print('length of minibatches:', len(mini_batches_train))
    
    model = LSTMLayers(batch=MINIBATCH)
    dtype = torch.FloatTensor
    
    number_of_workers = 8
    
    x_train = Variable(torch.from_numpy(x_train).type(dtype))
    y_train = Variable(torch.from_numpy(y_train).type(dtype))
    x_train_minibatch = DataLoader(x_train,
                                   batch_size=MINIBATCH,
                                   num_workers=number_of_workers,
                                   drop_last=True,
                                   pin_memory=True)
    
    y_train_minibatch = DataLoader(y_train,
                                   batch_size=MINIBATCH,
                                   num_workers=number_of_workers,
                                   drop_last=True,
                                   pin_memory=True)
    
    for i in range(len(mini_batches_train)):
#        x_train_batch, y_train_batch = mini_batches_train[i]
#        print('from for loop')
#        print(type(x_train_batch))
#        print(x_train_batch.shape)
#        x_train_batch = Variable(torch.from_numpy(
#                x_train_batch).type(dtype)).view(x_train_batch.shape[1], MINIBATCH, -1)
        x_train_batch = x_train_minibatch.contiguous().view(x_train_minibatch.shape[1], MINIBATCH, -1)
        print('type x_train_batch as Variable argument', type(x_train_batch))
        out = model(x_train_batch)
        print('print out from model:', out)
        
        
    

