
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



data=pd.read_csv("/Users/ziruilian/Desktop/RNN/topic3.csv")
data1 = data.to_numpy()

t2 = data1[5:466, 1]
t6=data1[5:466, 2]
t8=data1[5:466, 3]
t13=data1[5:466, 4]
t14=data1[5:466, 5]
t22=data1[5:466, 6]
t23=data1[5:466, 7]
t29=data1[5:466, 8]
t36=data1[5:466, 9]
t58=data1[5:466, 10]
t66=data1[5:466, 11]
t67=data1[5:466, 12]
t69=data1[5:466, 13]
t9=data1[5:466, 14]
t54=data1[5:466, 17]
t85=data1[5:466, 18]
ma=data1[4:465, 15]
price=data1[5:466, 16]

ts_t2=data1[466:496, 1]
ts_t6=data1[466:496, 2]
ts_t8=data1[466:496, 3]
ts_t13=data1[466:496, 4]
ts_t14=data1[466:496, 5]
ts_t22=data1[466:496, 6]
ts_t23=data1[466:496, 7]
ts_t29=data1[466:496, 8]
ts_t36=data1[466:496, 9]
ts_t58=data1[466:496, 10]
ts_t66=data1[466:496, 11]
ts_t67=data1[466:496, 12]
ts_t69=data1[466:496, 13]
ts_t9=data1[466:496, 14]
ts_t54=data1[466:496, 17]
ts_t85=data1[466:496, 18]
ts_ma=data1[465:495, 15]
actual_price=data1[466:496, 16]


#training data
TrainingInput=np.vstack((t2, t6, t8, t13, t14, t22, t23, t29, t36, t58, t66, t67, t69, t9, t54, t85, ma))
TrainingInput=np.transpose(TrainingInput)
trainingprice=[price]
TrainingOutput=np.transpose(trainingprice)
TestInput=np.vstack((ts_t2,ts_t6,ts_t8,ts_t13,ts_t14,ts_t22,ts_t23,ts_t29,ts_t36,ts_t58,ts_t66,ts_t67,ts_t69,ts_t9, ts_t54, ts_t85, ts_ma))
TestInput=np.transpose(TestInput)
Real_price=[actual_price]
real_price=np.transpose(Real_price)



np.shape(actual_price)


# In[128]:


#sklearn standardized
scaler1=MinMaxScaler()
MinMaxScaler(copy=True, feature_range=(0, 1))
Input=scaler1.fit_transform(TrainingInput)
Test_Input=scaler1.transform(TestInput)


# In[129]:


scaler2=MinMaxScaler()
MinMaxScaler(copy=True, feature_range=(0, 1))
Output=scaler2.fit_transform(TrainingOutput)


# In[130]:


input = np.array([Input]).reshape(461, 17)
input = input.astype(float)
target = np.array([Output]).reshape(461, 1)
target = target.astype(float)



# Create network with 2 layers
net = nl.net.newelm([[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]], [5, 1], [nl.trans.TanSig(), nl.trans.PureLin()])



# Train network
error = nl.train.train_rprop(net,input, target, epochs=10000, lr=0.000001, show=500, goal=0.0001)



# Simulate network
Test_Input = np.array([Test_Input]).reshape(30, 17)
Test_Input = Test_Input.astype(float)
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
    a = abs((x - y)/y)
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

