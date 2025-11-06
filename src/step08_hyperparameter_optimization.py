# -*- coding: utf-8 -*-
# step08_hyperparameter_optimization.py
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
import time, os, math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- evaluate one param combination ---
def _evaluate_model(model_name, base_model, param_dict, X_train, X_test, y_train, y_test, random_state=42):
    # apply parameters if given
    if param_dict:
        try:
            model = base_model.__class__(**param_dict)
        except TypeError:
            # fallback for models like LinearRegression with no hyperparams
            model = base_model
    else:
        model = base_model

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    result = {
        "model_name": model_name,
        "params": str(param_dict) if param_dict else "{}",
        "R2_train": r2_score(y_train, y_train_pred),
        "R2_test": r2_score(y_test, y_test_pred),
        "mae_train": mean_absolute_error(y_train, y_train_pred),
        "mae_test": mean_absolute_error(y_test, y_test_pred),
        "mse_train": mse_train,
        "mse_test": mse_test,
        "rmse_train": math.sqrt(mse_train),
        "rmse_test": math.sqrt(mse_test),
        "train_time_s": train_time
    }
    return result

# --- main parallel hyperparameter search ---
def run_param_search_parallel(X_train, X_test, y_train, y_test,
                              models_dict,
                              param_grids=None,
                              n_jobs=-1,
                              random_state=42,
                              results_path=None,
                              results_name="hyperparam_results.csv"):

    all_results = []

    for model_name, base_model in models_dict.items():
        # get param grid for this model
        grid = param_grids.get(model_name, None) if param_grids else None
        if grid:
            keys = list(grid.keys())
            vals = list(grid.values())
            param_combinations = [dict(zip(keys, combo)) for combo in product(*vals)]
        else:
            param_combinations = [None]  # single run with default params

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_evaluate_model)(model_name, base_model, params, X_train, X_test, y_train, y_test, random_state)
            for params in param_combinations
        )

        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    results_df["descriptors"] = [", ".join(X_train.columns)] * len(results_df)
    results_df = results_df.sort_values(by="R2_test", ascending=False).reset_index(drop=True)

    if results_path:
        os.makedirs(results_path, exist_ok=True)
        results_df.to_csv(os.path.join(results_path, results_name), index=False)

    # return best model
    top_row = results_df.iloc[0]
    best_model_name = top_row["model_name"]
    best_params = eval(top_row["params"])
    base_model = models_dict[best_model_name]

    try:
        best_model = base_model.__class__(**best_params)
    except TypeError:
        best_model = base_model

    best_model.fit(X_train, y_train)

    return results_df, top_row, best_model
