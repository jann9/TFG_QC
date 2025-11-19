import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy import stats  # For additional statistics

print("Starting model evaluation...")

class GCNRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNRegressor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Graph-level representation
        x = self.lin(x)
        return x


def load_graph_dataset(path,num_nodes):
    df = pd.read_csv(path)
    data_list = []
    print(f"Total rows in CSV: {len(df)}")
    if (num_nodes == "full"):
        num_nodes = 25
    for i, row in df.iterrows():
        q_start = 1
        expected_q_size = num_nodes * (num_nodes + 1) // 2
        q_end = q_start + expected_q_size
        print(num_nodes,expected_q_size)

        q_values = row.iloc[q_start:q_end].to_numpy(dtype=np.float32)

        if len(q_values) < expected_q_size:
            print(f"[Row {i}] Skipped: found {len(q_values)} Q values, expected {expected_q_size}")
            continue

        # Reconstruct full symmetric matrix
        q_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        idx = 0
        for u in range(num_nodes):
            for v in range(u, num_nodes):
                q_matrix[u, v] = q_values[idx]
                q_matrix[v, u] = q_values[idx]
                idx += 1

        src, dst = np.nonzero(q_matrix)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(q_matrix[src, dst], dtype=torch.float)

        x = torch.eye(num_nodes)  # One-hot node features

        label_start = q_end
        y_vals = row.iloc[label_start:].to_numpy(dtype=np.float32)
        y = torch.tensor(y_vals).unsqueeze(0)  # Shape (1, output_dim)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        data.num_nodes = num_nodes
        data_list.append(data)

    print(f"Loaded {len(data_list)} valid graphs.")
    return data_list


def compute_statistics(data, metric_name):
    """Compute comprehensive statistics for a given metric"""
    if len(data) == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'q1': np.nan,
            'q3': np.nan,
            'iqr': np.nan,
            'min': np.nan,
            'max': np.nan
        }
    
    return {
        'count': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'min': np.min(data),
        'max': np.max(data)
    }

def print_statistics(stats_dict, metric_name, f):
    """Print formatted statistics"""
    f.write(f"  {metric_name}:\n")
    f.write(f"    Count: {stats_dict['count']}\n")
    f.write(f"    Mean: {stats_dict['mean']:.6f}\n")
    f.write(f"    Median: {stats_dict['median']:.6f}\n")
    f.write(f"    Std Dev: {stats_dict['std']:.6f}\n")
    f.write(f"    Q1 (25%): {stats_dict['q1']:.6f}\n")
    f.write(f"    Q3 (75%): {stats_dict['q3']:.6f}\n")
    f.write(f"    IQR: {stats_dict['iqr']:.6f}\n")
    f.write(f"    Range: [{stats_dict['min']:.6f}, {stats_dict['max']:.6f}]\n")


def load_trained_gcn_model(model_path, num_nodes, output_dim, hidden_dim=32):
    """
    Load a trained GCN model
    
    Args:
        model_path: Path to the .pkl file
        num_nodes: Number of nodes (for input_dim)
        output_dim: Output dimension of the model
        hidden_dim: Hidden dimension (must match training)
    """
    # Create model with same architecture
    model = GCNRegressor(
        input_dim=num_nodes,  # Because we use one-hot encoding
        hidden_dim=hidden_dim, 
        output_dim=output_dim
    )
    
    # Load the saved state_dict
    model.load_state_dict(torch.load(model_path))
    
    # Set to evaluation mode
    model.eval()
    
    return model


def evaluate_loaded_model(model_path, dataset_path, num_nodes):
    """Evaluate a loaded GCN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    test_data = load_graph_dataset(dataset_path, num_nodes)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Determine output dimension from first sample
    output_dim = test_data[0].y.shape[1]
    
    # Load model
    model = load_trained_gcn_model(model_path, num_nodes, output_dim)
    model.to(device)
    
    # Evaluation
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return mse, mape, y_true, y_pred


# Your existing setup code here...
node_sizes = [10, 12, 15, 20, 25]
model_list = ['xgboost','MLP', 'GCN']
base_dataset_dir = "datasets"
output_file = "Models/Model_results.txt"

# === Start output file ===
with open(output_file, "w") as f:
    f.write("Model Evaluation Metrics\n")
    f.write("="*50 + "\n\n")

# Store all results for final summary
all_results = {}

for model_name in model_list:
    print(f"Evaluating {model_name}...")
    all_results[model_name] = {
        'individual': {'mse': [], 'mape': []},
        'full': {'mse': [], 'mape': []}
    }
    
    with open(output_file, "a") as f:
        f.write(f"\n{model_name}\n")
        f.write("-"*50 + "\n\n")
    
    # === Individual models ===
    individual_mse = []
    individual_mape = []
    
    for node_size in node_sizes:
        model_path = f"Models/{model_name}_model_{int(node_size)}.pkl"
        dataset_path = os.path.join(base_dataset_dir, f"dataset_{int(node_size)}_nodes.csv")

        # Error handling
        if not os.path.exists(model_path) or not os.path.exists(dataset_path):
            continue
            
        # Load model and dataset
        if model_name == 'GCN':
            mse, mape, _, _ = evaluate_loaded_model(model_path, dataset_path, node_size)
        else:
            model = joblib.load(model_path)
            df = pd.read_csv(dataset_path)

            # Extract columns and predict
            feature_cols = [col for col in df.columns if col.startswith("Q_")]
            target_cols = [col for col in df.columns if col.startswith("x_")]
            
            X = df[feature_cols].values
            y_true = df[target_cols].values
            y_pred = model.predict(X)
            
            mse = mean_squared_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
        
        individual_mse.append(mse)
        individual_mape.append(mape)
        
        with open(output_file, "a") as f:
            f.write(f"Node size {node_size:2d}: MSE = {mse:.6f}, MAPE = {mape:.6f}\n")

    # Store individual results
    all_results[model_name]['individual']['mse'] = individual_mse
    all_results[model_name]['individual']['mape'] = individual_mape

    # === Full Model ===
# === Full Model ===
    full_mse = []
    full_mape = []

    full_dataset_path = "datasets/dataset_full.csv"
    full_model_path = f"Models/{model_name}_model_full.pkl"

    if os.path.exists(full_model_path) and os.path.exists(full_dataset_path):
        # Add this check for GCN models
        if model_name == 'GCN':
            # Use 25 as the num_nodes for the full model (or read from first sample)
            mse, mape, _, _ = evaluate_loaded_model(full_model_path, full_dataset_path, 25)  # Changed from "full" to 25
            full_mse.append(mse)
            full_mape.append(mape)
            
            with open(output_file, "a") as f:
                f.write(f"\nFull Model Results:\n")
                f.write(f"  Full dataset: MSE = {mse:.6f}, MAPE = {mape:.6f}\n")
        else:
            # Original code for XGBoost and MLP
            model = joblib.load(full_model_path)
            df = pd.read_csv(full_dataset_path)
            feature_cols = [col for col in df.columns if col.startswith("Q_")]
            target_cols = [col for col in df.columns if col.startswith("x_")]
            node_sizes_full = sorted(df["num_nodes"].unique())
            
            with open(output_file, "a") as f:
                f.write(f"\nFull Model Results:\n")
            
            for node_size in node_sizes_full:
                df_subset = df[df["num_nodes"] == node_size]
                if df_subset.empty:
                    continue
                X = df_subset[feature_cols].values
                y_true = df_subset[target_cols].values
                y_pred = model.predict(X)
                
                mse = mean_squared_error(y_true, y_pred)
                mape = mean_absolute_percentage_error(y_true, y_pred)
                
                full_mse.append(mse)
                full_mape.append(mape)
                
                with open(output_file, "a") as f:
                    f.write(f"  num_nodes = {node_size:.6f}: MSE = {mse:.6f}, MAPE = {mape:.6f}\n")
    # Store full model results
    all_results[model_name]['full']['mse'] = full_mse
    all_results[model_name]['full']['mape'] = full_mape

# === COMPREHENSIVE FINAL STATISTICS ===
with open(output_file, "a") as f:
    f.write("\n" + "="*70 + "\n")
    f.write("COMPREHENSIVE STATISTICAL SUMMARY\n")
    f.write("="*70 + "\n\n")
    
    for model_name in model_list:
        f.write(f"{model_name.upper()} MODEL\n")
        f.write("-"*50 + "\n")
        
        # Individual models statistics
        f.write("Individual Models (Trained per Node Size):\n")
        mse_stats = compute_statistics(all_results[model_name]['individual']['mse'], 'MSE')
        mape_stats = compute_statistics(all_results[model_name]['individual']['mape'], 'MAPE')
        
        print_statistics(mse_stats, 'MSE', f)
        print_statistics(mape_stats, 'MAPE', f)
        
        # Full model statistics
        f.write("\nFull Model (Single Model for All Sizes):\n")
        mse_stats_full = compute_statistics(all_results[model_name]['full']['mse'], 'MSE')
        mape_stats_full = compute_statistics(all_results[model_name]['full']['mape'], 'MAPE')
        
        print_statistics(mse_stats_full, 'MSE', f)
        print_statistics(mape_stats_full, 'MAPE', f)
        f.write("\n")

# === COMPARATIVE ANALYSIS ===
with open(output_file, "a") as f:
    f.write("\n" + "="*70 + "\n")
    f.write("MODEL COMPARISON\n")
    f.write("="*70 + "\n\n")
    
    # Compare individual vs full models for each model type
    for model_name in model_list:
        f.write(f"{model_name.upper()}:\n")
        
        ind_mse_mean = np.mean(all_results[model_name]['individual']['mse'])
        full_mse_mean = np.mean(all_results[model_name]['full']['mse'])
        improvement = ((ind_mse_mean - full_mse_mean) / ind_mse_mean) * 100
        
        f.write(f"  MSE - Individual: {ind_mse_mean:.6f}, Full: {full_mse_mean:.6f}")
        f.write(f" (Improvement: {improvement:+.2f}%)\n")
        
        ind_mape_mean = np.mean(all_results[model_name]['individual']['mape'])
        full_mape_mean = np.mean(all_results[model_name]['full']['mape'])
        improvement_mape = ((ind_mape_mean - full_mape_mean) / ind_mape_mean) * 100
        
        f.write(f"  MAPE - Individual: {ind_mape_mean:.6f}, Full: {full_mape_mean:.6f}")
        f.write(f" (Improvement: {improvement_mape:+.2f}%)\n\n")

print(f"\nComprehensive evaluation results written to: {output_file}")