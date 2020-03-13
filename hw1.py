import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    #read data
    data = pd.read_csv('./train.csv', encoding = 'big5')
    data

    #preprocess
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()
    raw_data = raw_data.astype(np.float)
    for i in range (raw_data.shape[0]):
        times = 0
        sum = 0
        for j in range (raw_data.shape[1]):
            if (raw_data[i][j] >= 0):
                sum += raw_data[i][j]
                times += 1
        mean = (sum / times)
        for j in range (raw_data.shape[1]):
            if (raw_data[i][j] < 0):
                raw_data[i][j] = mean
    raw_data

    #將原始 4320 * 18 的資料依照每個月分重組成 12 個 18 (features) * 480 (hours) 的資料。
    month_data = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
        month_data[month] = sample

    #每個月會有 480hrs，每 9 小時形成一個 data，每個月會有 471 個 data，故總資料數為 471 * 12 筆，而每筆 data 有 9 * 18 的 features (一小時 18 個 features * 9 小時)。
    #對應的 target 則有 471 * 12 個(第 10 個小時的 PM2.5)
    x = np.empty([12 * 471, 18 * 9], dtype = float)
    y = np.empty([12 * 471, 1], dtype = float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
    print(x)
    print(y)

    #Normalize
    mean_x = np.mean(x, axis = 0) #18 * 9 
    std_x = np.std(x, axis = 0) #18 * 9 
    for i in range(len(x)): #12 * 471
        for j in range(len(x[0])): #18 * 9 
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    x

    #這部分是針對作業中report的第二題、第三題做的簡單示範，以生成比較中用來訓練的train_set和不會被放入訓練、只是用來驗證的validation_set。
    import math
    x_train_set = x[: math.floor(len(x) * 0.8), :]
    y_train_set = y[: math.floor(len(y) * 0.8), :]
    x_validation = x[math.floor(len(x) * 0.8): , :]
    y_validation = y[math.floor(len(y) * 0.8): , :]
    print(x_train_set)
    print(y_train_set)
    print(x_validation)
    print(y_validation)
    print(len(x_train_set))
    print(len(y_train_set))
    print(len(x_validation))
    print(len(y_validation))

    # 因為常數項的存在，所以 dimension (dim) 需要多加一欄；eps 項是避免 adagrad 的分母為 0 而加的極小數值。
    # 每一個 dimension (dim) 會對應到各自的 gradient, weight (w)，透過一次次的 iteration (iter_time) 學習。
    dim = 18 * 9 + 1
    w = np.zeros([dim, 1])
    x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)

    learning_rate = 5
    first = []
    iter_time = 300000
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
        if(t%100==0):
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight.npy', w)
    w

    # 載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，使 test data 形成 240 個維度為 18 * 9 + 1 的資料。
    # testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
    testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
    test_data = testdata.iloc[:, 2:]
    test_data[test_data == 'NR'] = 0
    test_data = test_data.to_numpy()
    test_x = np.empty([240, 18*9], dtype = float)
    for i in range(240):
        test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
    test_x

    w = np.load('weight.npy')
    ans_y = np.dot(test_x, w)
    ans_y

    import csv
    with open('submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)

