import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


node_sizes = [10, 12, 15, 20, 25]
model_list = ['xgboost','MLP']
base_dataset_dir = "datasets"
output_file = "Models/Model_results.txt"

# === Start output file ===
with open(output_file, "w") as f:
    f.write("Model Evaluation Metrics\n")
    f.write("="*40 + "\n\n")

# === Per edge probability loop ===
for model_name in model_list:
    print(model_name)
    with open(output_file, "a") as f:
        f.write("\n"+model_name+"\n")
        f.write("-"*40 + "\n\n")
    for node_size in node_sizes:
        # print(int(node_size))
        model_path = f"Models/{model_name}_model_{int(node_size)}.pkl"
        dataset_path = os.path.join(base_dataset_dir, f"dataset_{int(node_size)}_nodes.csv")

        # Error handling
        if not os.path.exists(model_path):
            with open(output_file, "a") as f:
                f.write(f" Model not found: {model_path}\n")
            continue
        # Load model
        model = joblib.load(model_path)

        # Load dataset
        if not os.path.exists(dataset_path):
            with open(output_file, "a") as f:
                f.write(f" Dataset not found: {dataset_path}\n")
            continue
        df = pd.read_csv(dataset_path)

        # Extract columns
        feature_cols = [col for col in df.columns if col.startswith("Q_")]
        target_cols = [col for col in df.columns if col.startswith("x_")]
        edge_probs = sorted(df["edge_prob"].unique())

        # Write node size
        with open(output_file, "a") as f:
            f.write(f"\n Node size: {node_size}\n")

        # Evaluate per edge probability
        for edge_prob in edge_probs:
            df_subset = df[df["edge_prob"] == edge_prob]
            if df_subset.empty:
                continue

            X = df_subset[feature_cols].values
            y_true = df_subset[target_cols].values
            y_pred = model.predict(X)
            mse = mean_squared_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            # Append result to file
            with open(output_file, "a") as f:
                f.write(f"   - edge_prob = {edge_prob:.1f}; MSE = {mse:.5f}, MAPE = {mape:.5f}\n")

    # === Full Model with padding ===

    full_dataset_path = "datasets/dataset_full.csv"
    full_model_path = f"Models/{model_name}_model_full.pkl"

    # Error handling
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f" Model not found: {full_model_path}")
    if not os.path.exists(full_dataset_path):
        raise FileNotFoundError(f" Dataset not found: {full_dataset_path}")

    # Load model and dataset 
    model = joblib.load(full_model_path)
    df = pd.read_csv(full_dataset_path)

    #  Extract columns
    feature_cols = [col for col in df.columns if col.startswith("Q_")]
    target_cols = [col for col in df.columns if col.startswith("x_")]

    # Detect unique node sizes in dataset
    node_sizes = sorted(df["num_nodes"].unique())
    #Write Full model
    with open(output_file, "a") as f:
            f.write(f"\n Full model:\n")
    # Evaluate model per node size ===
    for node_size in node_sizes:
        df_subset = df[df["num_nodes"] == node_size]
        if df_subset.empty:
            continue

        X = df_subset[feature_cols].values
        y_true = df_subset[target_cols].values
        y_pred = model.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
    
        with open(output_file, "a") as f:
            f.write(f" - num_nodes = {node_size}; MSE = {mse:.5f}, MAPE = {mape:.5f}\n")


print(f"\n Evaluation results written to: {output_file}")
