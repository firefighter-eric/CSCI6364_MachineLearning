import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def data_reprocess(filename):
    df = pd.read_csv(filename, header=0, sep=',')
    if 'label' in df:
        x = df.iloc[:, 1:]
        y = df.label
        return x.to_numpy(), y.to_numpy()
    else:
        x = df
        return x.to_numpy()


def scale(x_r, x):
    sc_x = StandardScaler()
    sc_x.fit(x_r)
    x = sc_x.transform(x)
    return x


def find_k(x_train, x_test, y_train, y_test, k_min, k_max):
    k_list = {}
    for k in range(k_min, k_max):
        print(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        s = knn.score(x_test, y_test)
        k_list[k] = s
        print("K:", k, "Score:", s)
    k_best = max(k_list, key=k_list.get)
    return k_best


def predict(x_data, y_data, x_predict, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_data, y_data)
    p = knn.predict(x_predict)
    return p


def write_file(filename, data):
    f = open(filename, 'w')
    i = 1
    f.write('ImageId,Label\n')
    for d in data:
        f.write(str(i) + ',' + str(d) + '\n')
        i += 1
    f.close()


if __name__ == '__main__':
    X_data, Y_data = data_reprocess('data_mnist.csv')

    # scale prepare
    sc_x = StandardScaler()
    sc_x.fit(X_data)
    X_data = sc_x.transform(X_data)
    print('preprocess end')

    # split
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data,
                                                        test_size=0.01, random_state=42, stratify=Y_data)
    print("split end")

    # # find k
    # K = find_k(X_train, X_test, Y_train, Y_test, 1, 2)
    # print("K:", K)

    # # predict
    # K = 3
    # X_predict = data_reprocess('test_mnist.csv')
    # X_predict = sc_x.transform(X_predict)
    # P = predict(X_data, Y_data, X_predict, K)
    # print('predict end')

    # # write file
    # write_file('out.txt', P)
    # print('finished')
