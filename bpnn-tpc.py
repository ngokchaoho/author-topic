import pandas as pd
import neurolab as nl
import numpy as np
import pylab as pl
from sklearn.preprocessing import MinMaxScaler

dataa = pd.read_csv("topics_bp_withemoji.csv", dtype=float)
data = dataa.to_numpy()
tn_price = data[0:461, 1]
tn_maa = data[0:461, 2]
inputt = data[0:461, 2:19]
ts_price = data[461:, 1]
ts_ma = data[461:, 2]
topicss = data[0:461, 3:19]
ts_topics = data[461:, 3:19]

s = len(ts_price)

tn_maa = np.reshape(tn_maa, (461, 1))
scaler1 = MinMaxScaler()
tn_ma = scaler1.fit_transform(tn_maa)

topics = np.reshape(topicss, (461, 16))
scaler_c = MinMaxScaler()
input_topic = scaler_c.fit_transform(topics)

outputt = np.reshape(tn_price, (461, 1))
scaler2 = MinMaxScaler()
output_data1 = scaler2.fit_transform(outputt)
output_data = (output_data1 - 0.5) / 0.5
min = np.amin(tn_ma)
max = np.amax(tn_ma)
minc = input_topic.min()
maxc = input_topic.max()

input_data = np.hstack((tn_ma, input_topic))

input_minmax = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
net = nl.net.newff(input_minmax, [5, 1])
net.trainf = nl.train.train_rprop
error = net.train(input_data, output_data, epochs=10000, show=1000, lr=0.000001, goal=0.05)

topics = data[0:, 3:17]
tn_price = tn_price.tolist()


ts_ma = np.reshape(ts_ma, (s, 1))
m = scaler1.transform(ts_ma)
ts_topics = np.reshape(ts_topics, (s, 16))
c = scaler_c.transform(ts_topics)
v = np.hstack((m, c))
v = np.reshape(v, (s, 17))
r = net.sim(v)
r1 = r / 2.0 + 0.5
r2 = scaler2.inverse_transform(r1)

ts_output = r2.tolist()



t = np.arange(1, s+1, 1)

result = ts_output
resultt = str(result)
resultt = resultt.replace('[', '')
resultt = resultt.replace(']', '')
resultt = resultt.replace('array', '')
resultt = resultt.replace('(', '')
resultt = resultt.replace(')', '')
result = list(eval(resultt))

pl.figure(1)
pl.plot(t, result, '-', t, ts_price, '.')
pl.legend(['predict price', 'actual price'])
pl.show()


def mse(x, y):
    return np.square(x - y).mean()


def mape(x, y):
    a = abs((x - y)/y)
    return a.mean()


t_mse = mse(ts_price, result)
t_mape = mape(result, ts_price)

diff_real = [None] * 31
diff_predict = [None] * 31

diff_real[0] = ts_price[0] - tn_price[460]
diff_predict[0] = result[0] - tn_price[460]

for i in range(1, s):
    diff_real[i] = ts_price[i] - ts_price[i - 1]
    diff_predict[i] = result[i] - result[i - 1]

direction_real = [None] * 31
direction_predict = [None] * 31

for i in range(0, s):
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

for i in range(0, s):
    if direction_real[i] == 1 and direction_predict[i] == 1:
        tp = tp + 1

    if direction_real[i] == 0 and direction_predict[i] == 1:
        fp = fp + 1

    if direction_real[i] == 1 and direction_predict[i] == 0:
        fn = fn + 1

precision = float(tp) / float(tp + fp)
recall = float(tp) / float(tp + fn)
f1score = 2 * (precision * recall) / (precision + recall)

