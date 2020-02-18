import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def data_preprocess(filename):
    df = pd.read_csv(filename, header=0, sep=',')
    x = df.iloc[:, 2:13]
    y = df.Salary
    return x.to_numpy(), y.to_numpy(), np.array(df.columns[2:13].tolist())


def scale(x_r, x):
    sc_x = StandardScaler()
    sc_x.fit(x_r)
    x = sc_x.transform(x)
    return x


def check_feature(x, y):
    reg = LinearRegression(n_jobs=-1).fit(x, y)
    score = reg.score(x, y)
    corr = reg.coef_ / np.std(y)
    select_list = [i for i, n in enumerate(corr) if abs(n) > 0.1]
    return score, corr, select_list


def train(x, y):
    reg = LinearRegression(n_jobs=-1).fit(x, y)
    score = reg.score(x, y)
    corr = reg.coef_ / np.std(y)
    return reg, score, corr


def valid(reg, x, y):
    score = reg.score(x, y)
    return score


if __name__ == '__main__':
    # preprocess
    X_raw, y_raw, Label = data_preprocess('baseball-9192.csv')
    X_data, y_data = scale(X_raw, X_raw), y_raw
    # std_x = 1
    std_y = np.std(y_data)

    Score, Corr, SelectList = check_feature(X_data, y_data)
    print(Label[SelectList])
    #
    X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data,
                                                          test_size=0.25, random_state=42)
    #
    Reg, ScoreTrain, CorrTrain = train(X_data[:, SelectList], y_data)
    ScoreValid = valid(Reg, X_valid[:, SelectList], y_valid)
