
import os
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


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
    bounds_vect = [[32,8,6]]
    #ml_config_bin = [ 0, 0, 0, 0, 0,0]
    # ml_config_bin = [1,1,1,1,1,1]
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

# Custom GNN model for regression
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

# Load dataset from CSV
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


def GNET_MLP(ml_config):
    base_dir = "datasets"  # Directory where datasets are stored
    output_file = "Models/Graph_Neural_Training_results.txt"
    metrics_csv = "Models/Graph_Neural_Training_metrics.csv"
    node_sizes = [10, 12, 15, 20, 25]  # Different dataset sizes

    # Prepare a dictionary to store results
    results_data = []

    # === Start output file ===
    with open(output_file, "w") as f:
        f.write("Model Training Metrics\n")
        f.write("="*40 + "\n\n")
    # Store the lossaverage of all datasets
    loss_average = []
    for num_nodes in node_sizes + ["full"]:
        dataset_filename = f"dataset_{num_nodes}_nodes.csv" if num_nodes != "full" else "dataset_full.csv"
        dataset_path = os.path.join(base_dir, dataset_filename)

        # Load dataset
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_filename} not found, skipping...")
            with open(output_file, "a") as f:
                f.write(f" Dataset not found: {dataset_path}\n")
            continue

        print(f"\n Training on {dataset_filename}...")
        # Load and split data
        dataset = load_graph_dataset(dataset_path,num_nodes)
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=41)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        # Write node size
        with open(output_file, "a") as f:
            f.write(f"\n Node size: {num_nodes}\n")

        # Setup model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        output_dim = dataset[0].y.shape[1]
        model = GCNRegressor(input_dim=dataset[0].num_nodes, hidden_dim=ml_config[0], output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # Training loop with timer
        start_time = time.time()
        for epoch in range(100):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.mse_loss(out, batch.y.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1:03d}, Loss: {total_loss/len(train_loader):.4f}")
        end_time = time.time()
        train_time = end_time - start_time

        # Evaluation + Metrics
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


        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        results_data.append({
            'node_size': num_nodes,
            'model_type': 'Q_values',
            'test_size': 0.2,
            'rmse': rmse,
            'mape': mape,
            'training_time': train_time,
            'model_file': f"GNN_model_{num_nodes}.pkl"
        })

        print(f"\nTest RMSE: {rmse:.4f}")
        print(f"Test MAPE: {mape:.4f}")
        print(f"Training time: {train_time:.2f} seconds")
        with open(output_file, "a") as f:
                f.write(f"   - RMSE = {rmse:.5f}, MAPE = {mape:.5f}, TIME = {train_time: .5f}\n")

        # Save the trained model as .pkl file
        model_save_path = f"Models/GCN_model_{num_nodes}.pkl"
        os.makedirs("Models", exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # Store the RNSE for each benchmark graph
        loss_average.append(rmse)

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(metrics_csv, index=False)
    print(f"\nMetrics saved to {metrics_csv}")

    print("\n **Final Training Results:**")
    print(results_df.to_string(index=False))

    print("\n Training complete for all datasets!")
    return np.average(loss_average)


def main():
    print("------------------------")
    print("Start of the You mother")
    print("------------------------")
    ml_config = read_solutions()
    start = time.time()
    results = GNET_MLP(ml_config)
    end = time.time()
    print(results)
    write_ftiness(results)
    print("------------------------")
    print("End of the Execution")
    print("------------------------")
    return;

if __name__ == "__main__":
    main()