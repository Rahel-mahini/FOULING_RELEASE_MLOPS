# -*- coding: utf-8 -*-

# step03_train_test_split.py

import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size=0.2, random_state=42):

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
