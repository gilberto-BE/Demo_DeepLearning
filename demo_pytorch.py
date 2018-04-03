#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 22:01:42 2018

@author: pia
"""

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

# =============================================================================
#     optimizer.step(closure)
#     Some optimization algorithms such as Conjugate Gradient and 
#     LBFGS need to reevaluate the function multiple times, 
#     so you have to pass in a closure that allows them to recompute your
#     model. The closure should clear the gradients, compute the loss, and return it.
# =============================================================================
    
    MINIBATCH = 2**4
    model = LSTMLayers(batch=MINIBATCH)
    dtype = torch.FloatTensor
    
    number_of_workers = 8
    
    x_train = torch.from_numpy(x_train).type(dtype)
    y_train = torch.from_numpy(y_train).type(dtype)
    
    x_train_minibatch = DataLoader(x_train,
                                   batch_size=MINIBATCH,
                                   num_workers=number_of_workers,
                                   drop_last=True)
    
    y_train_minibatch = DataLoader(y_train,
                                   batch_size=MINIBATCH,
                                   num_workers=number_of_workers,
                                   drop_last=True)
    model.train()
    model.hidden = model.init_hidden()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=0.001, 
                                 betas=(0.9, 0.999), 
                                 eps=1e-08, 
                                 weight_decay=0)
    
    for k, (input_data, target) in enumerate(
            zip(x_train_minibatch, y_train_minibatch)):
        def closure():
            model.zero_grad()
            features = input_data.view(input_data.size()[1], MINIBATCH, -1)
            features = Variable(features)
            label = Variable(target)
#            print('features')
#            print(features)
            out = model(features)
            loss = criterion(out, label)
            loss.backward()
            return loss
#            print(out)
        optimizer.step(closure)
            
        
        
    

