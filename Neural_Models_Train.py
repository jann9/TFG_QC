from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
import time
# Define the datasets to iterate over
node_sizes = [10, 12, 15, 20, 25]  # Different dataset sizes
base_dir = "datasets"  # Directory where datasets are stored
output_file = "Models/Neural_Training_results.txt"

# Prepare a dictionary to store results
results = {}

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1) # test size 0.25 default

    start_time = time.time() # Start timer
    # Define MLP model
    model = MLPRegressor(
        hidden_layer_sizes= (128, 64, 32),
        activation= "logistic",
        solver= "lbfgs",
        random_state=1, 
        max_iter=5000, 
        tol=0.1)

    # Train the model
    model.fit(X_train, y_train)
    end_time = time.time() # End timer
    train_time = end_time - start_time
    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    results[num_nodes, "mse"] = mse
    results[num_nodes, "mape"] = mape
    results[num_nodes, "time"] = train_time
    print(f" Mean Squared Error: {mse:.5f}")
    print(f" Mean Absolute Percentage Error: {mape:.5f}")
    print(f" Training time: {train_time: .5f}")
    with open(output_file, "a") as f:
            f.write(f"   - MSE = {mse:.5f}, MAPE = {mape:.5f}, TIME = {train_time: .5f}\n")
    
    # Save the trained model
    model_filename = f"Models/MLP_model_{num_nodes}.pkl"
    joblib.dump(model, model_filename)
    print(f" Model saved as {model_filename}")
    from sklearn.model_selection import GridSearchCV
'''
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (128, 64, 32)]
    }

    grid = GridSearchCV(MLPRegressor(max_iter=5000), param_grid, cv=3)
    grid.fit(X_train, y_train)
    print("Number of nodes;", num_nodes,"Best config:", grid.best_params_)
'''

print(results)
# Print final results summary
print("\n **Final MSE Results:**")
for (k, metric), value in results.items():
    print(f"Dataset {k}, {metric} = {value}")

print("\n Training complete for all datasets!")


