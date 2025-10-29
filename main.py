# main.py
import os
import yaml
import pandas as pd
from src import (
    generate_combinatorial_descriptors,
    split_train_test,
    normalize_features,
    select_k_best_features,
    build_model_from_config,
    train_eval_models,
    evaluate_model,
    save_model
)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config("config.yaml")
    
    # ----------------------------
    # Step 1: Generate combinatorial descriptors
    # ----------------------------

    print("Step 1: Generating combinatorial descriptors...")
    descriptors_path = generate_combinatorial_descriptors(config)
    
    # ----------------------------
    # Step 2: Train-test split
    # ----------------------------
    print("Step 2: Splitting train/test sets...")
    X_train, X_test, y_train, y_test = split_train_test(
        descriptors_path,
        config['data']['concentrations_file'],
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state']
    )
    
    # ----------------------------
    # Step 3: Preprocessing
    # ----------------------------
    print("Step 3: Preprocessing data...")
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)
    
    # ----------------------------
    # Step 4: Feature Selection
    # ----------------------------
    print("Step 4: Selecting top features...")
    top_1000_features = select_k_best_features(
        X_train_scaled, y_train, k=1000,
        output_file="outputs/top_1000_features.csv"
    )
    
    top_3_features = select_k_best_features(
        top_1000_features, y_train, k=3,
        output_file="outputs/top_3_features.csv"
    )
    
    # Select columns for training/testing
    X_train_selected = X_train_scaled[top_3_features.columns]
    X_test_selected = X_test_scaled[top_3_features.columns]
    
    # ----------------------------
    # Step 5: Model Training and evaluation
    # ----------------------------
    print("Step 5: Training models...")
    results_df, top_model = train_eval_models(X_train_selected, y_train, X_test_selected, y_test)

    # Save results as CSV
    os.makedirs(config['plots']['save_dir'], exist_ok=True)
    results_df.to_csv(os.path.join(config['plots']['save_dir'], 'models_results.csv'), index=False)

    print("Top model based on R2_test:")
    print(top_model)
    print(results_df)
    

    # ----------------------------
    # Step 6:  Update config dynamically and access the models
    # ----------------------------

    models_config = config['models_config']
    best_model_config = models_config[best_model_name]
    best_model = build_model_from_config(best_model_config)
    best_model_name = top_model['model_name']
    print(f" Selected best model: {best_model_name}")

    # ----------------------------
    # Step 6: Model Evaluation & Plots
    # ----------------------------
    print("Step 6: Evaluating models and generating plots...")

    descriptor_cols = list(top_model.iloc[2])
    evaluate_model(
    X_train=X_train_selected[descriptor_cols],
    X_test=X_test_selected[descriptor_cols],
    y_train=y_train,
    y_train_pred=best_model.predict(X_train_selected[descriptor_cols]),
    y_test=y_test,
    y_test_pred=best_model.predict(X_test_selected[descriptor_cols]),
    plots_dir="outputs/plots/",
    config={'model': best_model_config, 'plots': {'save_dir': 'outputs/plots/'}}
    )
    print("Pipeline completed successfully.")

   # ----------------------------
    # Step 7: Model serving
    # ----------------------------
    save_model(top_model, config)  # Serve the best model

if __name__ == "__main__":
    main()
