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
    visualize_model,
    save_model
)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config_path = os.path.join("config", "config.yaml")
    config = load_config("config_path")
    print(config)
    

    # Step 2: Generate combinatorial descriptors

    print("Step 2: Generating combinatorial descriptors...")
    descriptors_path = generate_combinatorial_descriptors(config)
    
   
    # Step 3: Train-test split
    
    print("Step 3: Splitting train/test sets...")
    
    X = pd.read_csv(descriptors_path])

    y = pd.read_csv(config['data']['target_file'])
    y = y.iloc[:, -1]
    y = y.squeeze()

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=config['models_config']['test_size'],
        random_state=config['models_config']['random_state']
    )
    
  
    # Step 4: Preprocessing
  
    print("Step 4: Preprocessing data...")
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)
    
   
    # Step 5: Feature Selection
   
    print("Step 5: Selecting top features...")

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
    
   
    # Step 6: Model Training and evaluation
    print("Step 5: Training and evaluating models...")
    results_df, top_model_row, best_model = train_eval_models(X_train_selected, y_train, X_test_selected, y_test)

    # Save results as CSV
    os.makedirs(config['plots']['save_dir'], exist_ok=True)
    results_df.to_csv(os.path.join(config['plots']['save_dir'], 'models_results.csv'), index=False)

    print("Top model based on R2_test:")
    print(top_model_row)
    print(results_df)
    
    
    # Step 7:  Update config dynamically and access the models
    
    best_model_name = top_model_row.iloc[0]
    print(f"Selected best model: {best_model_name}")
    
    models_config = config['models_config']
    best_model_config = models_config[best_model_name]


   
    # Step 8: Model Evaluation & Plots  
    print("Step 6: Evaluating models and generating plots...")



    descriptor_cols = list(top_model_row.iloc[2])
    visualize_model(
    X_train=X_train_selected[descriptor_cols],
    X_test=X_test_selected[descriptor_cols],
    y_train=y_train,
    y_train_pred=best_model.predict(X_train_selected[descriptor_cols]),
    y_test=y_test,
    y_test_pred=best_model.predict(X_test_selected[descriptor_cols]),
    save_dir =  config['plots']['save_dir'],
    config={'model': best_model_config, 'plots': config['plots']}
    )
    print("Pipeline completed successfully.")

    os.makedirs("outputs", exist_ok=True)

    # Step 9: Model serving
    save_model(best_model, "outputs/best_model.pkl")  # Serve the best model

if __name__ == "__main__":
    main()
