#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:30:01 2018

@author: Gilberto Batres-Estrada
"""

from IPython.display import Image
import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)
import OpenSSL.SSL
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt; plt.rcParams['figure.figsize'] = (10, 5)
import pylab
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, Imputer, QuantileTransformer

# This import is needed to suppress warnings, not sure why thera are warnings.
import warnings
warnings.simplefilter('error')

# Set the context: Run computation on cpus
data_ctx = mx.cpu() # data
model_ctx = mx.cpu() # parameters

from keras import optimizers
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import regularizers
#from sklearn.metrics import mean_squared_error



def to_return(x, period=1):
    """ This function supposes that the input is a
    dataframe or series"""
    return (x - x.shift(periods=period, axis='index'))/x.shift(periods=period)


def to_tensor(data, timesteps=30):
    x = np.array([data[i:i + timesteps]
                  for i in range(len(data) - timesteps)], dtype=float)
    return x



def split_train_dev_test(seq, dev=0.85, timesteps=30,
                         normalize=False, to_ret=False):

    x_train, x_dev, x_test = get_train_dev_test(seq, dev=dev)

#    # code added 2018-03-23
#    if to_ret:
#        x_train = x_train.apply(to_return)
#        x_dev = x_dev.apply(to_return)
#        x_test = x_test.apply(to_return)

    train_x, dev_x, test_x, scaler = transform_data(x_train,
                                                    x_dev, x_test,
                                                    normalize=normalize)

    train_x = to_tensor(train_x, timesteps=timesteps)
    x_train = train_x[:, :-1, :]
    y_train = train_x[:, -1]

    dev_x = to_tensor(dev_x, timesteps=timesteps)
    x_dev = dev_x[:, :-1, :]
    y_dev = dev_x[:, -1]

    test_x = to_tensor(test_x, timesteps=timesteps)
    x_test = test_x[:, :-1, :]
    y_test = test_x[:, -1]
    print('printing from the split-train-dev-test function:')
    print('y_dev raw data:')
    print(scaler.inverse_transform(y_dev))

    return x_train, y_train, x_dev, y_dev, x_test, y_test, scaler


def transform_data(x_train, x_dev, x_test, normalize=False):
    """ Do imputing and scaling in two steps. If done in
    pipeline then it is not possible to inverse-transform.
    No need to inverse transform imputing"""

    imputer = Imputer(strategy='mean')
    x_train = imputer.fit_transform(x_train)
    x_dev = imputer.transform(x_dev)
    x_test = imputer.transform(x_test)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_x = scaler.fit_transform(x_train)
    dev_x = scaler.transform(x_dev)
    test_x = scaler.transform(x_test)

    return train_x, dev_x, test_x, scaler


def get_train_dev_test(data_x, dev=0.85, drop_col=0.05):
    """ Split data in
    - training,
    - development
    - test data (also called live data)
    """
    """ We choose to keep the last 10 % of the data
    as test data (live trading, live data). The tests
    are performed when the model is finnished training"""

# =============================================================================
#     # Check if 10% or more are NAs if so drop those stocks
#     # inform how many are droped and how many are left.
#     dropped_stocks = []
#     for col in data_x.columns:
#         if data_x[col].isnull().sum()/len(data_x) > drop_col:
#             dropped_stocks.append(col)
#             data_x = data_x.drop(col, axis=1)
#             print('Stock {} has been dropped as it had more than {} % Nas'.\
#                   format(col, drop_col * 100))
# 
#     print('Number of stocks dropped:', len(dropped_stocks))
#     print('Number of stocks that are kept: ', len(data_x.columns))
# 
# =============================================================================
    test_idx = int(0.9 * len(data_x))
    x_test = data_x.iloc[test_idx:, :]

    """Get the first 90% of the data for train-dev"""
    train_dev_x = data_x.iloc[:test_idx]

    dev_idx = int(dev * len(train_dev_x))
    x_train = train_dev_x.iloc[:dev_idx, :]
    x_dev = train_dev_x.iloc[dev_idx:, :]

    return x_train, x_dev, x_test


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


def plot_error_curves(history):
    f1, axarr1 = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
    axarr1[0].plot(history.history['loss'])
    axarr1[0].set_title('Training Loss')
    axarr1[1].plot(history.history['val_loss'])
    axarr1[1].set_title('Dev Loss')
    axarr1[1].set_xlabel('Epochs')
#    f1.suptitle('MSE for stock: {}'.format(st_name))
plt.show()


if __name__ == '__main__':
    data = pd.read_csv('closing_prices_tiingo.csv', parse_dates=True, 
                       infer_datetime_format=True)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    print(data.head())
    
    plt.figure()
    data.plot(legend=False, title='All stocks in data set')
    plt.show()
    
    aapl = data['AAPL']
    print(aapl.head())
    
    
    plt.figure()
    aapl.plot(legend=True)
    plt.show()
    
    ret = to_return(aapl)
    print(ret.head())
    print('Shape of data: {}'.format(ret.shape))
    plt.figure()
    ret.plot(title='AAPL Returns')
    plt.show()
    
    
    plt.figure()
    stats.probplot(ret, dist='norm', plot=pylab)
    plt.title('QQ-plot for AAPL returns')
    pylab.show()
    
    
    x_train, y_train, x_dev, y_dev, x_test, y_test, scaler = split_train_dev_test(data)
    print('type of x_train', type(x_train[0]))
    print(x_train)
    print(x_train.shape)
    print(y_train.shape)
    print(x_dev.shape)
    print(y_dev.shape)
    
    
    
    num_features = x_train.shape[-1]
    print('num_features', num_features)
    num_outputs = y_train.shape[-1]
    print('num_outputs', num_outputs)
    drop_rate = 0.4
    lr = 0.001
    units = 100
    epochs = 150
    decay = 1e-6
    look_back = x_train.shape[1]
    batch_size = 64

    optim = optimizers.Adam(lr=lr,
                            beta_1=0.9,
                            beta_2=0.999,
                            decay=0.0,
                            clipnorm=1.0)

    ret_seq = False
    model = Sequential()
    model.add(LSTM(units,input_shape=(look_back, num_features),
#                   return_sequences=ret_seq,
                   kernel_regularizer=regularizers.l2(0.01),
                   recurrent_regularizer=regularizers.l2(0.01),
                   activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(num_outputs, kernel_regularizer=regularizers.l2(0.01)))
    model.compile(loss='mean_squared_error',
                  optimizer=optim,
                  metrics=['mse'])

    history = model.fit(x_train, y_train, epochs, batch_size,
                        verbose=0, validation_data=(x_dev, y_dev),
                        shuffle=False)

    train_predict = model.predict(x_train)
    dev_predict = model.predict(x_dev)

    mse_train = mean_squared_error(y_train, train_predict)
    mse_dev = mean_squared_error(y_dev, dev_predict)

    plot(history)