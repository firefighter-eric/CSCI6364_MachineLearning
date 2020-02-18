import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree


def data_preprocess(filename):
    df = pd.read_csv(filename, header=0, sep=',')
    x = df.iloc[:, 2:-1].fillna(method='ffill', axis='columns')
    y = df.Legendary
    return x.to_numpy(), y.to_numpy(), df.columns[2:-1]


if __name__ == '__main__':
    X_data, y_data, Label = data_preprocess('Pokemon.csv')

    # one hot code
    enc_1 = OneHotEncoder(handle_unknown='ignore').fit(X_data[:, [0]])
    Feature1_one_hot = enc_1.transform(X_data[:, [0]]).toarray()
    enc_2 = OneHotEncoder(handle_unknown='ignore').fit(X_data[:, [1]])
    Feature2_one_hot = enc_2.transform(X_data[:, [1]]).toarray()
    X_one_hot = (Feature1_one_hot + Feature2_one_hot) > 0.5
    Feature_num = X_one_hot.sum(1).reshape((-1, 1))
    X_data_2 = np.concatenate((X_one_hot, Feature_num, X_data[:, 2:]), axis=1)
    # split
    X_train, X_test, y_train, y_test = train_test_split(X_data_2, y_data,
                                                        test_size=0.25, random_state=42)
    # train
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    print('1')
    clf = clf.fit(X_train, y_train)
    s_train = clf.score(X_train, y_train)
    s_test = clf.score(X_test, y_test)
    tree_label = enc_1.categories_[0].tolist() + ['Feature Num'] + Label[2:].tolist()
    # tree_label = [i for i in range(1, len(X_one_hot[0]) + 1)] + Label[2:].tolist()
    r = tree.export_text(clf, feature_names=tree_label, max_depth=10)

    print(r)
    print('Train Score:', s_train, 'Valid Score:', s_test)
