import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
import time

# Define the datasets to iterate over
node_sizes = [10, 12, 15, 20, 25]  # Different dataset sizes
base_dir = "datasets"  # Directory where datasets are stored
output_file = "Models/XGB_Training_results.txt"
metrics_csv = "Models/XGB_Training_metrics.csv" 

# Prepare a dictionary to store results
results_data = []

# === Start output file ===
with open(output_file, "w") as f:
    f.write("Model Training Metrics\n")
    f.write("="*40 + "\n\n")
    
# Iterate over each dataset (one per node size)
for num_nodes in node_sizes + ["full"]:  # Also include the full dataset
    dataset_filename = f"dataset_{num_nodes}_nodes.csv" if num_nodes != "full" else "dataset_full.csv"
    dataset_path = os.path.join(base_dir, dataset_filename)

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_filename} not found, skipping...")
        with open(output_file, "a") as f:
            f.write(f" Dataset not found: {dataset_path}\n")
        continue
    
    print(f"\n Training on {dataset_filename}...")

    df = pd.read_csv(dataset_path)

    # Extract features (Q values) and target (x values)
    feature_columns = [col for col in df.columns if col.startswith("Q_")]
    output_columns = [col for col in df.columns if col.startswith("x_")]
    
    # Write node size
    with open(output_file, "a") as f:
        f.write(f"\n Node size: {num_nodes}\n")
        
    # print(df.columns)
    X = df[feature_columns].values
    y = df[output_columns].values

    # Split into training and testing data (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time() # Start timer
    # Define XGBoost model
    model = xgb.XGBRegressor(
        objective="reg:squarederror",  
        n_estimators=500,  # More trees
        max_depth=4,  # Reduce depth to avoid overfitting
        learning_rate=0.05,  # Slower learning
        subsample=0.8,  # Use 80% of data per tree
        colsample_bytree=0.8,  # Use 80% of features per tree
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)
    end_time = time.time() # End timer
    train_time = end_time - start_time
    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    
    results_data.append({
        'node_size': num_nodes,
        'model_type': 'Q_values',
        'dataset_size': len(df),
        'n_features': len(feature_columns),
        'n_outputs': len(output_columns),
        'rmse': rmse,
        'mape': mape,
        'training_time': train_time,
        'model_file': f"xgboost_model_{num_nodes}.pkl"
    })
    
    
    print(f" Root Mean Squared Error: {rmse:.5f}")
    print(f" Mean Absolute Percentage Error: {mape:.5f}")
    print(f" Training time: {train_time: .5f}")
    with open(output_file, "a") as f:
            f.write(f"   - RMSE = {rmse:.5f}, MAPE = {mape:.5f}, TIME = {train_time: .5f}\n")
    # Save the trained model
    model_filename = f"Models/xgboost_model_{num_nodes}.pkl"
    joblib.dump(model, model_filename)
    print(f" Model saved as {model_filename}")
    
    
    # ==== SECOND MODEL: Circuit features ====
    dataset_filename = f"dataset_{num_nodes}_nodes_Circuit.csv" if num_nodes != "full" else "dataset_full_Circuit.csv"
    dataset_path = os.path.join(base_dir, dataset_filename)

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_filename} not found, skipping...")
        with open(output_file, "a") as f:
            f.write(f" Dataset not found: {dataset_path}\n")
        continue
    
    print(f"\n Training on {dataset_filename}...")

    df = pd.read_csv(dataset_path)

    # Extract features (Q values) and target (x values)
    feature_columns = ['depth', 'num_qubits', 'rx', 'rz', 'cx']
    output_columns = [col for col in df.columns if col.startswith("x_")]
    
    # Write node size
    with open(output_file, "a") as f:
        f.write(f"\n Node size: {num_nodes} Circuit\n")
        
    # print(df.columns)
    X = df[feature_columns].values
    y = df[output_columns].values

    # Split into training and testing data (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time() # Start timer
    # Define XGBoost model
    model = xgb.XGBRegressor(
        objective="reg:squarederror",  
        n_estimators=500,  # More trees
        max_depth=4,  # Reduce depth to avoid overfitting
        learning_rate=0.05,  # Slower learning
        subsample=0.8,  # Use 80% of data per tree
        colsample_bytree=0.8,  # Use 80% of features per tree
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)
    end_time = time.time() # End timer
    train_time = end_time - start_time
    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    results_data.append({
        'node_size': num_nodes,
        'model_type': 'Circuit',
        'dataset_size': len(df),
        'n_features': len(feature_columns),
        'n_outputs': len(output_columns),
        'rmse': rmse,
        'mape': mape,
        'training_time': train_time,
        'model_file': f"xgboost_model_{num_nodes}_Circuit.pkl"
    })
    
    print(f" Root Mean Squared Error: {rmse:.5f}")
    print(f" Mean Absolute Percentage Error: {mape:.5f}")
    print(f" Training time: {train_time: .5f}")
    with open(output_file, "a") as f:
            f.write(f"   - RMSE = {rmse:.5f}, MAPE = {mape:.5f}, TIME = {train_time: .5f}\n")
    # Save the trained model
    model_filename = f"Models/xgboost_model_{num_nodes}_Circuit.pkl"
    joblib.dump(model, model_filename)
    print(f" Model saved as {model_filename}")


# ===== Test full models on individual node test sets =====
print("\n=== Testing Full Models on Individual Node Test Sets ===")

# Test Q-values full model on each node size subset from the full dataset
full_q_model_path = "Models/xgboost_model_full.pkl"
full_dataset_path = os.path.join(base_dir, "dataset_full.csv")

if os.path.exists(full_q_model_path) and os.path.exists(full_dataset_path):
    full_q_model = joblib.load(full_q_model_path)
    print("Full Q-values model loaded successfully")
    
    # Load the full dataset
    full_df = pd.read_csv(full_dataset_path)
    
    for num_nodes in node_sizes:  # [10, 12, 15, 20, 25]
        print(f"\nTesting full Q-values model on {num_nodes} nodes subset...")
        
        # Filter the full dataset for specific num_nodes
        df_subset = full_df[full_df['num_nodes'] == num_nodes].copy()
        
        if len(df_subset) == 0:
            print(f"No data found for {num_nodes} nodes in full dataset")
            continue
        
        # Extract features and targets
        feature_columns = [col for col in df_subset.columns if col.startswith("Q_")]
        output_columns = [col for col in df_subset.columns if col.startswith("x_")]
        
        X = df_subset[feature_columns].values
        y = df_subset[output_columns].values
        
        # Use the SAME random_state as during training to get identical test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        # Predict using the full model
        y_pred = full_q_model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Add to results
        results_data.append({
            'node_size': num_nodes,
            'model_type': 'Q_values_full_model',
            'dataset_size': len(df_subset),
            'n_features': len(feature_columns),
            'n_outputs': len(output_columns),
            'rmse': rmse,
            'mape': mape,
            'training_time': 0,  # No training time for testing
            'model_file': full_q_model_path
        })
        
        print(f"  RMSE on {num_nodes} nodes subset: {rmse:.5f}")
        print(f"  MAPE on {num_nodes} nodes subset: {mape:.5f}")
        
        with open(output_file, "a") as f:
            f.write(f"\nFull Q-values model tested on {num_nodes} nodes subset:\n")
            f.write(f"   - RMSE = {rmse:.5f}, MAPE = {mape:.5f}\n")

# Circuit models don't need per-node testing - only Q-values models do
results_df = pd.DataFrame(results_data)
results_df.to_csv(metrics_csv, index=False)
print(f"\nMetrics saved to {metrics_csv}")
# Print final results summary
print("\n **Final Training Results:**")
# Print final results summary
print("\n **Final Training Results:**")
print(results_df.to_string(index=False))


print("\n Training complete for all datasets!")