import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os

# Define the datasets to iterate over
node_sizes = [10, 12, 15, 20, 25]  # Different dataset sizes
base_dir = "datasets"  # Directory where datasets are stored

# Prepare a dictionary to store results
results = {}

# Iterate over each dataset (one per node size)
for num_nodes in node_sizes + ["full"]:  # Also include the full dataset
    dataset_filename = f"dataset_{num_nodes}_nodes.csv" if num_nodes != "full" else "dataset_full.csv"
    dataset_path = os.path.join(base_dir, dataset_filename)

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_filename} not found, skipping...")
        continue
    
    print(f"\n🔹 Training on {dataset_filename}...")

    df = pd.read_csv(dataset_path)

    # Extract features (Q values) and target (x values)
    feature_columns = [col for col in df.columns if col.startswith("Q_")]
    output_columns = [col for col in df.columns if col.startswith("x_")]
    # print(df.columns)
    X = df[feature_columns].values
    y = df[output_columns].values

    # Split into training and testing data (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    results[num_nodes, "mse"] = mse
    results[num_nodes, "mape"] = mape
    print(f" Mean Squared Error: {mse:.5f}")
    print(f" Mean Absolute Percentage Error: {mape:.5f}")

    # Save the trained model
    model_filename = f"Models/xgboost_model_{num_nodes}.pkl"
    joblib.dump(model, model_filename)
    print(f" Model saved as {model_filename}")

print(results)
# Print final results summary
print("\n **Final MSE Results:**")
for (k, metric), value in results.items():
    print(f"Dataset {k}, {metric} = {value}")

print("\n Training complete for all datasets!")
