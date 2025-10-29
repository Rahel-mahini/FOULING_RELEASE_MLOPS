# -*- coding: utf-8 -*-
"""
Evaluation and visualization: correlation plots, Williams, ALE
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from numpy.linalg import pinv
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from alibi.explainers import ALE, plot_ale

def evaluate_model(X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, plots_dir):
    def correlation_plot(y_train, y_train_pred, y_test, y_test_pred, save_path='correlation_plot.png'):
        plt.figure(figsize=(8,6))
        plt.scatter(y_train, y_train_pred, color='skyblue', label='Training', alpha=0.7, s=100)
        plt.scatter(y_test, y_test_pred, color='#007FDE', label='Test', alpha=0.7, s=100)
        plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='k', linestyle='--', linewidth=2)
        plt.xlabel('Experimental')
        plt.ylabel('Predicted')
        plt.legend()
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        plt.text(0.95, 0.05, f'R² Train: {r2_train:.3f}\nR² Test: {r2_test:.3f}', 
                horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

    def williams_plot(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, save_path='williams_plot.png'):
        H_train = X_train @ pinv(X_train.T @ X_train) @ X_train.T
        leverage_train = np.diag(H_train)
        H_test = X_test @ pinv(X_test.T @ X_test) @ X_test.T
        leverage_test = np.diag(H_test)
        std_res_train = (y_train - y_train_pred)/np.std(y_train - y_train_pred)
        std_res_test = (y_test - y_test_pred)/np.std(y_test - y_test_pred)
        
        n_train, p = len(y_train), X_train.shape[1]
        threshold_leverage = 3*(p+1)/n_train
        threshold_resid = 3
        
        plt.figure(figsize=(8,8))
        plt.scatter(leverage_train, std_res_train, alpha=0.6, s=100, label='Training', color='skyblue')
        plt.scatter(leverage_test, std_res_test, alpha=0.6, s=100, label='Test', color='#007FDE')
        plt.axhline(y=threshold_resid, color='k', linestyle='--')
        plt.axhline(y=-threshold_resid, color='k', linestyle='--')
        plt.axvline(x=threshold_leverage, color='k', linestyle='--')
        plt.xlabel('Leverage')
        plt.ylabel('Standardized Residuals')
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.show()


    def evaluate_model_ale(X_train, y_train, config):
        """
        Fit a model and generate ALE plots for interpretability.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series or np.array): Training targets
            config (dict): YAML configuration containing model parameters and output paths
        """
        warnings.filterwarnings("ignore")
        
        # ---------------- Normalize features ----------------
        scaler = MinMaxScaler()
        X_train_normalized = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

        # ---------------- Initialize model ----------------
            # ---------------- Initialize model dynamically ----------------
        model_params = config.get('model', {})
        model_type = model_params.get('type', 'DecisionTreeRegressor')  # default
        model_mapping = {
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'LinearRegression': LinearRegression,
            'Lasso': Lasso,
            'Ridge': Ridge,
            'RandomForest': RandomForestRegressor,
            'SVR': SVR
        }

        if model_type not in model_mapping:
            raise ValueError(f"Model type {model_type} not implemented.")
        
        ModelClass = model_mapping[model_type]

        # Extract parameters dynamically
        # Remove 'type' key from params
        model_kwargs = {k: v for k, v in model_params.items() if k != 'type'}
        if 'random_state' not in model_kwargs:
            model_kwargs['random_state'] = 42  # default random_state if applicable
        
        model = ModelClass(**model_kwargs)

        
        # ---------------- Fit the model ----------------
        model.fit(X_train_normalized, y_train)

        # ---------------- Compute ALE ----------------
        feature_names = list(X_train_normalized.columns)
        target_name = config.get('plots', {}).get('target_name', 'Target')
        gb_ale = ALE(model.predict, feature_names=feature_names, target_names=[target_name])
        
        ale_exp = gb_ale.explain(X_train_normalized.values)

        # ---------------- Plot ALE ----------------
        save_dir = config.get('plots', {}).get('save_dir', 'outputs/plots/')
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 9))
        plot_ale(ale_exp, ax=ax)
        ax.set_title('', fontsize=16)
        ax.set_xlabel(feature_names[0], fontsize=18)  # Adjust if plotting multiple features
        ax.set_ylabel('ALE Value', fontsize=18)
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        plt.legend(title=target_name, fontsize=14, title_fontsize=16, loc='best')

        save_path = os.path.join(save_dir, f'ALE_plot_{feature_names[0]}.png')
        plt.savefig(save_path, dpi=600, transparent=True)
        plt.show()

        print(f"ALE plot saved at: {save_path}")
        return ale_exp