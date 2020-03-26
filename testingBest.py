import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

input = sys.argv[1]
output = sys.argv[2]

features = 14
totMonths = 11
testdata = pd.read_csv(input, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data = test_data.to_numpy()
test_data[test_data == 'NR'] = 0
test_data = test_data.astype(np.float)
for i in range (test_data.shape[0]):
    times = 0
    sum = 0
    for j in range (test_data.shape[1]):
        if (test_data[i][j] >= 0):
            sum += test_data[i][j]
            times += 1
    mean = (sum / times)
    for j in range (test_data.shape[1]):
        if (test_data[i][j] < 0):
            test_data[i][j] = mean
test_data

# drop features
test_data = pd.DataFrame(test_data)
print(test_data)
for i in range(240):
    test_data = test_data.drop([5 + 18 * i, 10 + 18 * i, 13 + 18 * i, 15 + 18 * i], axis = 0)
test_data = test_data.values

test_x = np.empty([240, features*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[features * i: features* (i + 1), :].reshape(1, -1)

#Normalize
mean_x = np.load('mean_xBest.npy')
std_x = np.load('std_xBest.npy')
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
test_x

# 有了 weight 和測試資料即可預測 target。
w = np.load('weightBest.npy')
ans_y = np.dot(test_x, w)
ans_y[ans_y < 0] = 0
ans_y


with open(output, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)