import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

input = sys.argv[1]
output = sys.argv[2]

features = 18
totMonths = 12
testdata = pd.read_csv(input, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data = test_data.to_numpy()
test_data[test_data == 'NR'] = 0
test_data = test_data.astype(np.float)

test_x = np.empty([240, features*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[features * i: features* (i + 1), :].reshape(1, -1)

#Normalize
mean_x = np.load('mean_x.npy')
std_x = np.load('std_x.npy')
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
test_x

# 有了 weight 和測試資料即可預測 target。
w = np.load('weight.npy')
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