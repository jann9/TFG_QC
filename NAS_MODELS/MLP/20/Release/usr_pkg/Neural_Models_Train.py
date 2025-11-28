
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
import time
import sys
import random
from numpy import random


def write_ftiness(metric):
    fitness_file = open("fitness.txt", "w")
    fitness_file.write(str(metric))
    fitness_file.close()
    return;

def read_solutions():
    file = open('individual.txt', 'r')
    ml_config_bin = []
    while 1:
        # read by character
        component = file.read(1)
        if not component: 
            break
        ml_config_bin.append(int(component))
    file.close()
    # ------ convert to int -------
    bounds_vect = [[224,32,8],[112,16,7],[56,8,6], [8750,1250,14]]
    #ml_config_bin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]
    # ml_config_bin = [1,1,1,1,1,1,1,1,  1,1,1,1,1,1,1,  1,1,1,1,1,1,  1,1,1,1,1,1,1,1,1,1,1,1,1]
    ml_config_int_vect = []
    lower_bound_vect = 0
    for bounds in bounds_vect: 
        sup_index = bounds[2]
        upper_bound_vect = lower_bound_vect + sup_index
        shift = 2**(sup_index) - 1 - (bounds[0] - bounds[1])
        ml_config_int = ml_config_bin[lower_bound_vect:upper_bound_vect]
        int_value = 0
        for  i in range (0,sup_index-1):
            int_value += 2**(i)*ml_config_bin[lower_bound_vect+i]
        int_value += bounds[1] + (2**(sup_index-1)- shift)*ml_config_bin[upper_bound_vect-1]
        ml_config_int_vect.append(int_value)
        lower_bound_vect += sup_index
    return ml_config_int_vect;



def read_solutions_p3(ml_config_bin):
    # ------ convert to int -------
    bounds_vect = [[224,32,8],[112,16,7],[56,8,6], [8750,1250,14]]
    #ml_config_bin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]
    # ml_config_bin = [1,1,1,1,1,1,1,1,  1,1,1,1,1,1,1,  1,1,1,1,1,1,  1,1,1,1,1,1,1,1,1,1,1,1,1]
    ml_config_int_vect = []
    lower_bound_vect = 0
    for bounds in bounds_vect:
        sup_index = bounds[2]
        upper_bound_vect = lower_bound_vect + sup_index
        shift = 2**(sup_index) - 1 - (bounds[0] - bounds[1])
        ml_config_int = ml_config_bin[lower_bound_vect:upper_bound_vect]
        int_value = 0
        for  i in range (0,sup_index-1):
            int_value += 2**(i)*ml_config_bin[lower_bound_vect+i]
        int_value += bounds[1] + (2**(sup_index-1)- shift)*ml_config_bin[upper_bound_vect-1]
        ml_config_int_vect.append(int_value)
        lower_bound_vect += sup_index
    return ml_config_int_vect;


def NAS_MLP(ml_config,node_sizes):
    # Define the datasets to iterate over
    base_dir = "datasets"  # Directory where datasets are stored
    output_file = f"Models/Neural_nas_Training_results_{node_sizes[0]}_nodes.txt"

    # Prepare a dictionary to store results
    results_data = []

    # === Start output file ===
    with open(output_file, "w") as f:
        f.write("Model Training Metrics\n")
        f.write("="*40 + "\n\n")

    # Iterate over each dataset (one per node size)
    for num_nodes in node_sizes:  # Also include the full dataset
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

        X = df[feature_columns].values
        y = df[output_columns].values

        # Split into training and testing data (80%-20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        start_time = time.time() # Start timer
        # Define MLP model
        model = MLPRegressor(
            hidden_layer_sizes= (ml_config[0], ml_config[1], ml_config[2]),
            activation= "logistic",
            solver= "lbfgs",
            random_state=random.randint(100),
            max_iter=ml_config[3],
            learning_rate_init=0.002,
            tol=0.1)

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
            'hidden_layers': str((ml_config[0], ml_config[1], ml_config[2])),
            'activation': 'logistic',
            'solver': 'lbfgs',
            'test_size': 0.2,
            'rmse': rmse,
            'mape': mape,
            'training_time': train_time,
            'converged': model.n_iter_ < model.max_iter,
            'n_iterations': model.n_iter_,
            'n_iteration_max': ml_config[3],
            'model_file': f"MLP_model_nas_{num_nodes}.pkl"
        })



        """
        # The circuit model has been discarded for the moment
        # CIRCUIT MODELS
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

        X = df[feature_columns].values
        y = df[output_columns].values

        # Split into training and testing data (80%-20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        start_time = time.time() # Start timer
        # Define MLP model
        model = MLPRegressor(
            hidden_layer_sizes= (ml_config[0], ml_config[1], ml_config[2]),
            activation= "logistic",
            solver= "lbfgs",
            random_state=1,
            max_iter=ml_config[3],
            tol=0.1)

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
            'hidden_layers': str((ml_config[0], ml_config[1], ml_config[2])),
            'activation': 'logistic',
            'solver': 'lbfgs',
            'test_size': 0.2,
            'rmse': rmse,
            'mape': mape,
            'training_time': train_time,
            'converged': model.n_iter_ < model.max_iter,
            'n_iterations': model.n_iter_,
            'model_file': f"MLP_model_nas_{num_nodes}_Circuit.pkl"
        })

        print(f" Root Mean Squared Error: {rmse:.5f}")
        print(f" Mean Absolute Percentage Error: {mape:.5f}")
        print(f" Training time: {train_time: .5f}")
        with open(output_file, "a") as f:
                f.write(f"   - RMSE = {rmse:.5f}, MAPE = {mape:.5f}, TIME = {train_time: .5f}\n")

        # Save the trained model
        model_filename = f"Models/MLP_model_nas_{num_nodes}_Circuit.pkl"
        joblib.dump(model, model_filename)
        print(f" Model saved as {model_filename}")
        """

    """
    # The circuit model has been discarded for the moment
    # ===== Test full models on individual node test sets =====
    print("\n=== Testing Full Models on Individual Node Test Sets ===")

    # Test Q-values full model on each node size subset from the full dataset
    full_q_model_path = "Models/MLP_model_nas_full.pkl"
    full_dataset_path = os.path.join(base_dir, "dataset_full.csv")

    if os.path.exists(full_q_model_path) and os.path.exists(full_dataset_path):
        full_q_model = joblib.load(full_q_model_path)
        print("Full Q-values model loaded successfully")

        # Load the full dataset
        full_df = pd.read_csv(full_dataset_path)
        # Store the lossaverage of all datasets
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
                'hidden_layers': str((ml_config[0], ml_config[1], ml_config[2])),
                'activation': 'logistic',
                'solver': 'lbfgs',
                'test_size': 0.2,
                'rmse': rmse,
                'mape': mape,
                'training_time': 0,  # No training time for testing
                'converged': True,
                'n_iterations': 0,
                'model_file': full_q_model_path
            })

            print(f"  RMSE on {num_nodes} nodes subset: {rmse:.5f}")
            print(f"  MAPE on {num_nodes} nodes subset: {mape:.5f}")

            with open(output_file, "a") as f:
                f.write(f"\nFull Q-values model tested on {num_nodes} nodes subset:\n")
                f.write(f"   - RMSE = {rmse:.5f}, MAPE = {mape:.5f}\n")
            loss_average.append(rmse)

    # Circuit models don't need per-node testing - only Q-values models do

    """
    return rmse,mape,train_time,model,results_data


def main():
    print("------------------------")
    print("Start of the Execution")
    print("------------------------")
    node_sizes=[20]
    metrics_csv = f"Models/MLP_nas_Training_metrics_{node_sizes[0]}_nodes.csv"
    output_file = f"Models/Neural_nas_Training_results_{node_sizes[0]}_nodes.txt"
    base_dir_solutions="results"
    solutions_filename="solutions.dat"
    solutions_path = os.path.join(base_dir_solutions, solutions_filename)
    if not os.path.exists(solutions_path):
        f = open("fitness.txt")
        previous_fitness = float(f.read())
    else:
        with open(solutions_path, 'r') as f:
            last_line = f.readlines()[-1]
        x = last_line.split(" ")
        config=[]
        for char in x[2]:
            if char != "\n":
                config.append(int(char))
        previous_fitness= -1*float(x[0])
        previous_config=read_solutions_p3(config)
    ml_config = read_solutions()
    results = NAS_MLP(ml_config,node_sizes)
    write_ftiness(results[0])
    rmse=float(results[0])
    mape=results[1]
    train_time=results[2]
    model=results[3]
    results_data=results[4]
    if rmse < previous_fitness:
        # Write node size
        with open(output_file, "a") as f:
            f.write(f"\n Node size: {node_sizes[0]}\n")
        print(f" Root Mean Squared Error: {rmse:.5f}")
        print(f" Mean Absolute Percentage Error: {mape:.5f}")
        print(f" Training time: {results[2]: .5f}")
        with open(output_file, "a") as f:
                f.write(f"   - RMSE = {rmse:.5f}, MAPE = {mape:.5f}, TIME = {train_time: .5f}\n")
        # Save the trained model
        model_filename = f"Models/MLP_model_nas_{node_sizes[0]}.pkl"
        joblib.dump(model, model_filename)
        print(f" Model saved as {model_filename}")

        # Save all results to CSV
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(metrics_csv, index=False)
        print(f"\nMetrics saved to {metrics_csv}")

        # Print final results summary
        print("\n **Final Training and Cross-Testing Results:**")
        print(results_df.to_string(index=False))
        print("\n Training and cross-testing complete for all datasets!")
    print("------------------------")
    print("End of the Execution")
    print("------------------------")
    return;

if __name__ == "__main__":
    main()