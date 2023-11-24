import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data_creditcard(val_split=True):
    np.random.seed(11235)
    df = pd.read_csv("../data/creditcard_2023.csv")
    X = df.drop(["id", "Class"], axis=1)
    y = df[["Class"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if val_split:
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test)
        X_val = sc.transform(X_val)
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test
