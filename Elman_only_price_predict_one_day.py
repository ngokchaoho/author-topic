import numpy as np
import pandas as pd
import neurolab as nl
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plot
import pylab as pl
import numpy as np
dtype=np.float64


data=pd.read_csv("/Users/ziruilian/Desktop/RNN/topic2.csv")
data1 = data.to_numpy()

#training data
ma=data1[4:465, 15]
price=data1[5:466, 16]

ts_ma=data1[465:495, 15]
actual_price=data1[466:496, 16]

#training data
training_input=[ma]
Training_Input=np.transpose(training_input)

training_output=[price]
Traing_Output=np.transpose(training_output)

#testing data
Test_input=[ts_ma]
TestInput=np.transpose(Test_input)
real_price=[actual_price]
real_price=np.transpose(real_price)


#standardization of training data
scaler1 = MinMaxScaler()
input = scaler1.fit_transform(Training_Input)
input = input.astype(float)

Test_Input=scaler1.transform(TestInput)

scaler2 = MinMaxScaler()
target = scaler2.fit_transform(Traing_Output)
target = target.astype(float)

input = np.array([input]).reshape(461, 1)
input = input.astype(float)
target = np.array([target]).reshape(461, 1)
target = target.astype(float)

Test_Input = np.array([Test_Input]).reshape(30, 1)
Test_Input = Test_Input.astype(float)

# Create network with 2 layers
net = nl.net.newelm([[0, 1]], [5, 1], [nl.trans.TanSig(), nl.trans.PureLin()])



# Train network
error = nl.train.train_rprop(net,input, target, epochs=10000, lr=0.000001, show=500, goal=0.0001)


# Simulate network
outputt = net.sim(Test_Input)

Output=scaler2.inverse_transform(outputt)

pl.figure(1)

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
pl.plot(x, Output, '-', x, real_price,'+-')
pl.legend(['predict price', 'real price'])

pl.show()


def mse(x, y):
    return np.mean(np.square(x - y))


def mape(x, y):
    a = abs((x - y) / y)
    return a.mean()


diff_real = [None] * 31
diff_predict = [None] * 31

diff_real[0] = Output[0] - real_price[0]

for i in range(0, 30):
    diff_real[i] = real_price[i] - real_price[i - 1]

for i in range(0, 30):
    diff_predict[i] = Output[i] - Output[i - 1]

direction_real = [None] * 31
direction_predict = [None] * 31

for i in range(0, 30):
    if diff_real[i] >= 0:
        direction_real[i] = 1
    else:
        direction_real[i] = 0

    if diff_predict[i] >= 0:
        direction_predict[i] = 1
    else:
        direction_predict[i] = 0

tp = 0
fp = 0
fn = 0

for i in range(0, 30):
    if direction_real[i] == 1 and direction_predict[i] == 1:
        tp = tp + 1

    if direction_real[i] == 0 and direction_predict[i] == 1:
        fp = fp + 1

    if direction_real[i] == 1 and direction_predict[i] == 0:
        fn = fn + 1

precision = float(tp) / float(tp + fp)
recall = float(tp) / float(tp + fn)
f1score = 2 * (precision * recall) / (precision + recall)

result = np.asarray(Output)
Real_Price = np.asarray(real_price)

t_mse = mse(Real_Price, result)
t_mape = mape(Real_Price, result)

