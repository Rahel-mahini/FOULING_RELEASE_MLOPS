# -*- coding: utf-8 -*-
# step07_model_train_eval.py
import os
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def loo_r2_score(model, X_train, y_train):
    """Compute LOO R2"""
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X_train):
        # Use .iloc for row selection
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_hat = model.predict(X_train.iloc[test_idx])
        y_true.append(y_train.iloc[test_idx].values[0])
        y_pred.append(y_hat[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return r2_score(y_true, y_pred)

def select_k_best_features(X, y, k, output_file):
    """Select top k features using RandomForest importance"""
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
    grid.fit(X, y)
    best_rf = grid.best_estimator_
    importances = best_rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:k]
    top_features = X.iloc[:, top_idx]
    top_features.to_csv(output_file, index=False)
    return top_features

def train_eval_models(X_train, y_train, X_test, y_test):
    """Train models on top 3 features combinations and return results_df sorted by R2_test"""
    results_df = pd.DataFrame()

    models = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(alpha=1.0),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest_default': RandomForestRegressor(random_state=42),
        'RandomForest_simple': RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
        'DecisionTree_default': DecisionTreeRegressor(random_state=42),
        'DecisionTree_simple': DecisionTreeRegressor(max_depth=3, random_state=42),
        'SVR_linear': SVR(kernel='linear'),
        'SVR_rbf': SVR(kernel='rbf')
    }

    selected_features = X_train.columns.tolist()

    # Loop over all combinations of top 3 features
    for j in range(1, 4):
        for combo in combinations(selected_features, j):
            X_train_sel = X_train[list(combo)]
            X_test_sel = X_test[list(combo)]

            for name, model in models.items():
                model.fit(X_train_sel, y_train)
                y_train_pred = model.predict(X_train_sel)
                y_test_pred = model.predict(X_test_sel)

                R2_train = r2_score(y_train, y_train_pred)
                R2_test = r2_score(y_test, y_test_pred)
                mae_train = mean_absolute_error(y_train, y_train_pred)
                mae_test = mean_absolute_error(y_test, y_test_pred)
                mse_train = mean_squared_error(y_train, y_train_pred)
                mse_test = mean_squared_error(y_test, y_test_pred)
                rmse_train = np.sqrt(mse_train)
                rmse_test = np.sqrt(mse_test)

                # LOO R2
                loo_r2 = loo_r2_score(model, X_train_sel, y_train)

                results_df = pd.concat([
                    results_df,
                    pd.DataFrame({
                        'model_name': [name],
                        'num_features': [j],
                        'descriptors': [combo],
                        'R2_train': [R2_train],
                        'R2_test': [R2_test],
                        'mae_train': [mae_train],
                        'mae_test': [mae_test],
                        'mse_train': [mse_train],
                        'mse_test': [mse_test],
                        'rmse_train': [rmse_train],
                        'rmse_test': [rmse_test],
                        'loo_r2': [loo_r2]
                    })
                ], ignore_index=True)

    # Sort by R2_test and return top 3 models
    results_df = results_df.sort_values(by='R2_test', ascending=False).reset_index(drop=True)
    top_model_row = results_df.iloc[1]

    top_model_name = top_model_row.iloc[0]
    print(f"Selected best model: {top_model_name}")
    top_features = list(top_model_row.iloc[2])

    # Re-train the best model on the full training set
    best_model = models[top_model_name]
    best_model.fit(X_train[top_features], y_train)


    return  results_df, top_model_row, best_model
