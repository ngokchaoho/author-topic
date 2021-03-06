# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:38:19 2019

@author: Daniela Du
"""
#Source code: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/?spm=a2c4e.11153940.blogcont174270.18.76893039Fzbuny


# Input X are the 5 days moving average of Bitcoin price at a given time (t)，d2，d6,d8, d12, d17, d22, d23, d29, d36, d58, d66, d67, d69, d93
# Output Y is the 5 days moving average of Bitcoin price the next time (t + 1).

import os
import pandas as pd
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
os.chdir("E:\Master program\Courses\Semester 2\FE5225 Machine Learning\Project\My part\Withemojis")

## load the dataset
#dataset = pd.read_csv('data1.csv', header=0, index_col=0)
#def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#	n_vars = 1 if type(data) is list else data.shape[1]
#	df = pd.DataFrame(data)
#	cols, names = list(), list()
#	# input sequence (t-n, ... t-1)
#	for i in range(n_in, 0, -1):
#		cols.append(df.shift(i))
#		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#	# forecast sequence (t, t+1, ... t+n)
#	for i in range(0, n_out):
#		cols.append(df.shift(-i))
#		if i == 0:
#			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#		else:
#			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#	# put it all together
#	agg = pd.concat(cols, axis=1)
#	agg.columns = names
#	# drop rows with NaN values
#	if dropnan:
#		agg.dropna(inplace=True)
#	return agg

values = dataset.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
#reframed = series_to_supervised(scaled, 1, 1)
#
## drop columns we don't want to predict
#reframed.drop(reframed.columns[14:16], axis=1, inplace=True)
#reframed.drop(reframed.columns[14:28], axis=1, inplace=True)
#print(reframed.head())
values = scaled.values
train = scaled[:460, :]
test = scaled[460:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# fit network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)



# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


# invert scaling for forecast
inv_yhat = concatenate((test_X,yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
inv_yhat = pd.DataFrame(inv_yhat)
test_y = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_X,test_y), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
#inv_y = inv_y[:,0]

true_data = dataset.iloc[460:,-1].reset_index()
true_data.drop('Date',axis = 1,inplace=True)
true_data.columns = [0]
fig1 = plt.figure(1)
plt.plot(inv_yhat, 'r:')
plt.plot(true_data, 'g-')
plt.legend(['Predict', 'True'])
plt.show()


inv_yhat1 = inv_yhat.copy()
# make predictions
#trainPredict = model.predict(train_X)
#train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
#inv_yhat_train = concatenate((trainPredict, train_X), axis=1)
#inv_yhat_train = scaler.inverse_transform(inv_yhat_train)
#inv_yhat_train = inv_yhat_train[:,0]
#train_y = train_y.reshape((len(train_y), 1))
#inv_y_train = concatenate((train_y, train_X), axis=1)
#inv_y_train = scaler.inverse_transform(inv_y_train)
#inv_y_train = inv_y_train[:,0]
#
#dataset.reset_index(inplace=True)
#dataset = dataset.drop('Date',axis = 1)
#fig2 = plt.figure(2)
#plt.plot(inv_yhat_train, 'r:')
#plt.plot(dataset.iloc[:460,-1], 'g-')
#plt.legend(['predict_train', 'true'])
#plt.show()
    
# calculate parameters
mse = (mean_squared_error(true_data, inv_yhat))
#mse = (mean_squared_error(inv_y_train, inv_yhat_train))
mape = np.mean(np.abs((true_data - inv_yhat) / true_data)) * 100

TP = 0
FP = 0
FN = 0
for i in range(1,len(true_data)):
    if true_data.iloc[i,0]-true_data.iloc[i-1,0]>0 and inv_yhat.iloc[i,0]-inv_yhat.iloc[i-1,0] >0:
        TP = TP+1
    elif true_data.iloc[i,0]-true_data.iloc[i-1,0]<0 and inv_yhat.iloc[i,0]-inv_yhat.iloc[i-1,0] >0:
        FP = FP+1
    elif true_data.iloc[i,0]-true_data.iloc[i-1,0]>0 and inv_yhat.iloc[i,0]-inv_yhat.iloc[i-1,0] <0:
        FN = FN+1
   
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1_score = 2*precision*recall/(precision+recall)