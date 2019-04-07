import pandas as pd
import neurolab as nl
import numpy as np
import pylab as pl
from sklearn.preprocessing import MinMaxScaler

dataa = pd.read_csv("topics_bp_withemoji.csv", dtype=float)
data = dataa.to_numpy()
tn_price1 = data[0:461, 1].tolist()
tn_ma = data[0:461, 2].tolist()
ts_price = data[461:, 1].tolist()
ts_ma = data[461:, 2].tolist()

s = len(ts_price)


tn_ma = np.reshape(tn_ma, (461, 1))
scaler1 = MinMaxScaler()
input_data = scaler1.fit_transform(tn_ma)


tn_price = np.reshape(tn_price1, (461, 1))
scaler2 = MinMaxScaler()

output_data1 = scaler2.fit_transform(tn_price)
output_data = (output_data1 - 0.5) / 0.5


nett = nl.net.newff([(0, 1)], [5, 1])
nett.trainf = nl.train.train_rprop
error = nett.train(input_data, output_data, epochs=10000, show=1000, lr=0.000001, goal=0.05)

#tn_price = tn_price.tolist()

ts_ma = np.reshape(ts_ma, (s, 1))

n = scaler1.transform(ts_ma)
r = nett.sim(n)
n1 = np.reshape(r, (s, 1))
n2 = n1 * 0.5 + 0.5
ts_output = scaler2.inverse_transform(n2)

ts_output = ts_output.tolist()

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
    return np.mean(np.square(x - y))


def mape(x, y):
    a = abs((x - y)/y)
    return a.mean()


diff_real = [None] * 31
diff_predict = [None] * 31

b = str(ts_output)
b = b.replace('[', '')
b = b.replace(']', '')
b = b.replace('array', '')
b = b.replace('(', '')
b = b.replace(')', '')
a = list(eval(b))

diff_real[0] = ts_price[0] - tn_price1[460]
diff_predict[0] = a[0] - tn_price1[460]

for i in range(1, s):
    diff_real[i] = ts_price[i] - ts_price[i - 1]
    diff_predict[i] = a[i] - a[i - 1]

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

accuracy = (float(tp) + float(fn)) / s
print(accuracy)
precision = float(tp) / float(tp + fp)
recall = float(tp) / float(tp + fn)
f1score = 2 * (precision * recall) / (precision + recall)

result = np.asarray(ts_output)
ts_price = np.asarray(ts_price)

t_mse = mse(ts_price, result)
t_mape = mape(result, ts_price)