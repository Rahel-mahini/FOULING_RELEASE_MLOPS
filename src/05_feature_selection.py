# -*- coding: utf-8 -*-
"""
Feature selection using Random Forest importance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def select_k_best_features(X_train, y_train, k=100, output_file=None):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10],
        'min_samples_split': [10],
        'min_samples_leaf': [2],
        'max_features': ['log2'],
        'bootstrap': [False]
    }
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    
    best_rf = grid.best_estimator_
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1][:k]
    
    selected_features = X_train.iloc[:, indices]
    if output_file:
        selected_features.to_csv(output_file, index=False)
    return selected_features
