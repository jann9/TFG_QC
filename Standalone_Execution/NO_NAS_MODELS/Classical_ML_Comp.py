import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import numpy as np
import os 
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# from qiskit.optimization.applications.ising import max_cut
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_aer import Aer
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from PIL import Image
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from qiskit_ibm_runtime import SamplerV2 as Sampler
import random
import time
import sys


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
    
    
node_sizes = [10, 12, 15, 20, 25]
edge_probs = [0.5, 0.6, 0.7, 0.8, 0.9]
model_list = ['xgboost', 'MLP']
num_graphs_per_combination = 10
base_dataset_dir = "datasets"

df = pd.read_csv("datasets/maxcut_dataset.csv")

def extract_graph(num_nodes, probability, graph_index):
    """Extracts a specific graph from the dataset and returns a NetworkX graph."""
    sub_df = df[(df["num_nodes"] == num_nodes) & 
                (df["probability"] == probability) & 
                (df["graph_index"] == graph_index)]
    
    G = nx.Graph()
    for _, row in sub_df.iterrows():
        G.add_edge(int(row["node_1"]), int(row["node_2"]), weight=row["weight"])
    
    return G





# --------------------------- Create the Hamiltonian


#   Convert Graph into QUBO matrix

def maxcut_qubo(G):
    # Generate QUBO matrix for MaxCut problem, mapping nodes to indices.
    node_list = list(G.nodes())  # Get list of nodes
    # node_index = {node: idx for idx, node in enumerate(node_list)}  # Map nodes to indices
    n = len(node_list)  # Number of nodes

    Q = np.zeros((n, n))  # Initialize QUBO matrix

    for i, j in G.edges():
        w = G[i][j]['weight']
        # idx_i = node_index[i]  # Convert node to index
        # idx_j = node_index[j]  # Convert node to index
        Q[i, i] -= w
        Q[j, j] -= w
        Q[i, j] += 2 * w  # Off-diagonal term

    return Q, node_list  # Return QUBO and node mapping



def get_pauli_list(Q):
    # Convert QUBO matrix to a PauliSumOp Hamiltonian for QAOA.
    num_vars = Q.shape[0]  # Number of qubits
    pauli_terms = []

    for i in range(num_vars):
        for j in range(num_vars):
            if Q[i, j] != 0 and i != j:
                # Create an identity string of length num_vars
                pauli_str = ["I"] * num_vars  
                
                # Apply Pauli-Z operators at positions i and j
                pauli_str[i], pauli_str[j] = "Z", "Z"

                # Convert to string format and add the term
                pauli_terms.append(("".join(pauli_str)[::-1], Q[i, j]))

    return pauli_terms  # Construct Pauli Hamiltonian


print("\n")






#-------------------------- Optimization of the parameters

backend = Aer.get_backend('qasm_simulator')
# print(backend)

# Create pass manager for transpilation
pm = generate_preset_pass_manager(optimization_level=3,
                                    backend=backend)




# Initial parameters
initial_gamma = np.pi
initial_beta = np.pi/2
init_params = [initial_gamma, initial_beta, initial_gamma, initial_beta]


def cost_func_estimator(params, ansatz, hamiltonian, estimator):

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    objective_func_vals.append(cost)


    return cost





objective_func_vals = [] # Global variable

    
    
def extract_upper_triangular(Q):
    return Q[np.triu_indices(Q.shape[0])]
def padding_extract_upper_triangular(Q, max_size):
    upper_tri = Q[np.triu_indices(Q.shape[0])]
    pad_size = (max_size * (max_size + 1)) // 2 - len(upper_tri)
    return np.pad(upper_tri, (0, pad_size), mode='constant')

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


def evaluate_maxcut_solution(G, solution):
    """
    Evaluate the Max-Cut value for a given solution
    
    Args:
        G: NetworkX graph with edge weights
        solution: Binary solution vector (or continuous values that will be rounded)
    
    Returns:
        cut_value: Sum of weights of edges in the cut
    """
    # Round continuous solutions to binary
    binary_solution = np.round(solution)
    
    cut_value = 0
    for edge in G.edges():
        u, v = edge
        # If nodes are in different partitions, add edge weight to cut
        if binary_solution[u] != binary_solution[v]:
            weight = G[u][v].get('weight', 1.0)  # Get weight, default to 1.0
            cut_value += weight
    
    return cut_value

def get_optimal_maxcut(G):
    """
    Get the optimal Max-Cut value using brute force (only for small graphs)
    or a good approximation for larger graphs
    
    Args:
        G: NetworkX graph with edge weights
    
    Returns:
        optimal_cut: Best weighted cut value found
    """
    n = G.number_of_nodes()
    
    # For small graphs, use brute force
    if n <= 15:
        best_cut = 0
        for i in range(2**n):
            # Generate binary string
            solution = [(i >> j) & 1 for j in range(n)]
            cut_value = evaluate_maxcut_solution(G, solution)
            best_cut = max(best_cut, cut_value)
        return best_cut
    else:
        # For larger graphs, use a weighted greedy approximation
        partition = [0] * n
        for node in range(n):
            # Count WEIGHTED edges to each partition
            weight_to_0 = sum(G[node][neighbor].get('weight', 1.0) 
                            for neighbor in G.neighbors(node) 
                            if partition[neighbor] == 0)
            weight_to_1 = sum(G[node][neighbor].get('weight', 1.0) 
                            for neighbor in G.neighbors(node) 
                            if partition[neighbor] == 1)
            # Assign to partition with fewer weighted connections
            partition[node] = 0 if weight_to_0 <= weight_to_1 else 1
        
        return evaluate_maxcut_solution(G, partition)

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


def generate_dataset_with_timing_and_fitness(node_sizes, edge_probs, num_graphs_per_combination, base_dir="datasets"):
    """
    Modified version that captures both timing and fitness (solution quality) data
    """
    os.makedirs(base_dir, exist_ok=True)
    max_size = max(node_sizes)
    all_data = []
    
    # Dictionary to store results
    results = {
        'node_sizes': [],
        'classical_times': [],
        'classical_cuts': [],
        'optimal_cuts': [],
        'model_times': {model: [] for model in model_list},
        'model_cuts': {model: [] for model in model_list}
    }
   
    for num_nodes in node_sizes:
        data = []
        prob = random.choice(edge_probs)
        graph_index = random.choice(range(num_graphs_per_combination))
        G = extract_graph(num_nodes=num_nodes, probability=prob, graph_index=graph_index)
        Q, node_mapping = maxcut_qubo(G)
        
        # Get optimal/best known cut value
        optimal_cut = get_optimal_maxcut(G)
        results['optimal_cuts'].append(optimal_cut)
        
        # === Classical QAOA ===
        start_time = time.time()
        paulis = get_pauli_list(Q)
        H = SparsePauliOp.from_list(paulis)
        QAOAcircuit = QAOAAnsatz(cost_operator=H, reps=2)
        QAOAcircuit.measure_all()
        candidate_circuit = pm.run(QAOAcircuit)
       
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 1000
            result = minimize(
                cost_func_estimator,
                init_params,
                args=(candidate_circuit, H, estimator),
                method="COBYLA",
                tol=1e-2,
            )
            optimal_params = result.x
            
            # Now sample from the circuit with optimal parameters to get bitstrings
            sampler = Sampler(mode=session)
            sampler.options.default_shots = 1000
            
            # Bind parameters to circuit
            bound_circuit = candidate_circuit.assign_parameters(optimal_params)
            
            # Sample the circuit
            job = sampler.run([bound_circuit])
            result_sampler = job.result()
            
            # Get counts from the result
            counts = result_sampler[0].data.meas.get_counts()
            
            # Get the most frequent bitstring
            best_bitstring = max(counts, key=counts.get)
            
            # Convert bitstring to numpy array (reverse for qiskit's little-endian)
            x_classical = np.array([int(bit) for bit in reversed(best_bitstring)])

        end_time = time.time()
        classical_test_time = end_time - start_time

        # Evaluate classical solution quality
        classical_cut = evaluate_maxcut_solution(G, x_classical)        
        print(f"Classical {num_nodes} nodes - Time: {classical_test_time:.4f}s, Cut: {classical_cut}/{optimal_cut}")
        
        results['node_sizes'].append(num_nodes)
        results['classical_times'].append(classical_test_time)
        results['classical_cuts'].append(classical_cut)
        
        # === ML Models ===
        for model_name in model_list:
            start_time = time.time()
            print(str(sys.argv[1]))
            model_path = f"Models/ml_vs_ml/{model_name}/Execution_{str(sys.argv[1])}/Models/{model_name}_model_{int(num_nodes)}.pkl"
            
            
            # Load model and predict QAOA parameters
            model = joblib.load(model_path)
            
            # Flatten Q matrix (upper triangular) as input
            q_flat = []
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    q_flat.append(Q[i, j])
            
            X = np.array(q_flat).reshape(1, -1)
            predicted_params = model.predict(X)[0]  # This should be [gamma1, beta1, gamma2, beta2]
            
            # Now run QAOA circuit with ML-predicted parameters (no optimization!)
            with Session(backend=backend) as session:
                sampler = Sampler(mode=session)
                sampler.options.default_shots = 1000
                
                # Use the same circuit as classical QAOA
                bound_circuit = candidate_circuit.assign_parameters(predicted_params)
                
                # Sample the circuit
                job = sampler.run([bound_circuit])
                result_sampler = job.result()
                
                # Get counts from the result
                counts = result_sampler[0].data.meas.get_counts()
                
                # Get the most frequent bitstring
                best_bitstring = max(counts, key=counts.get)
                
                # Convert bitstring to numpy array
                x_model = np.array([int(bit) for bit in reversed(best_bitstring)])
            
            end_time = time.time()
            model_test_time = end_time - start_time
            
            # Evaluate ML-predicted solution quality
            model_cut = evaluate_maxcut_solution(G, x_model)
            
            print(f"{model_name} {num_nodes} nodes - Time: {model_test_time:.4f}s, Cut: {model_cut}/{optimal_cut}")
            
            results['model_times'][model_name].append(model_test_time)
            results['model_cuts'][model_name].append(model_cut)
    
    # Create comparison visualizations
    #create_comprehensive_comparison(results)
    create_comprehensive_comparison_V2(results)
    return results

def create_comprehensive_comparison(results, save_prefix="comparison"):
    """
    Creates comprehensive comparison graphs for both timing and fitness
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    node_sizes = results['node_sizes']
    classical_times = results['classical_times']
    classical_cuts = results['classical_cuts']
    optimal_cuts = results['optimal_cuts']
    
    colors = ['blue', 'green', 'purple', 'orange', 'brown']
    
    # === Plot 1: Time Comparison (Linear Scale) ===
    ax1 = axes[0, 0]
    ax1.plot(node_sizes, classical_times, 'o-', linewidth=2, markersize=8, 
             label='Classical QAOA', color='red', alpha=0.8)
    
    for i, (model_name, times) in enumerate(results['model_times'].items()):
        ax1.plot(node_sizes, times, 's--', linewidth=2, markersize=8,
                label=f'{model_name}', color=colors[i % len(colors)], alpha=0.8)
    
    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Time Comparison (Log Scale) ===
    ax2 = axes[0, 1]
    ax2.plot(node_sizes, classical_times, 'o-', linewidth=2, markersize=8, 
             label='Classical QAOA', color='red', alpha=0.8)
    
    for i, (model_name, times) in enumerate(results['model_times'].items()):
        ax2.plot(node_sizes, times, 's--', linewidth=2, markersize=8,
                label=f'{model_name}', color=colors[i % len(colors)], alpha=0.8)
    
    ax2.set_xlabel('Number of Nodes', fontsize=12)
    ax2.set_ylabel('Time (seconds, log scale)', fontsize=12)
    ax2.set_title('Execution Time Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # === Plot 3: Cut Quality (Absolute Values) ===
    ax3 = axes[1, 0]
    ax3.plot(node_sizes, optimal_cuts, 'k*-', linewidth=3, markersize=12,
             label='Optimal/Best Known', zorder=10, alpha=0.9)
    ax3.plot(node_sizes, classical_cuts, 'o-', linewidth=2, markersize=8, 
             label='Classical QAOA', color='red', alpha=0.8)
    
    for i, (model_name, cuts) in enumerate(results['model_cuts'].items()):
        ax3.plot(node_sizes, cuts, 's--', linewidth=2, markersize=8,
                label=f'{model_name}', color=colors[i % len(colors)], alpha=0.8)
    
    ax3.set_xlabel('Number of Nodes', fontsize=12)
    ax3.set_ylabel('Cut Value', fontsize=12)
    ax3.set_title('Solution Quality: Max-Cut Values', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # === Plot 4: Approximation Ratio ===
    ax4 = axes[1, 1]
    
    # Calculate approximation ratios
    classical_ratios = [c/o if o > 0 else 0 for c, o in zip(classical_cuts, optimal_cuts)]
    ax4.plot(node_sizes, classical_ratios, 'o-', linewidth=2, markersize=8, 
             label='Classical QAOA', color='red', alpha=0.8)
    
    for i, (model_name, cuts) in enumerate(results['model_cuts'].items()):
        model_ratios = [c/o if o > 0 else 0 for c, o in zip(cuts, optimal_cuts)]
        ax4.plot(node_sizes, model_ratios, 's--', linewidth=2, markersize=8,
                label=f'{model_name}', color=colors[i % len(colors)], alpha=0.8)
    
    ax4.axhline(y=1.0, color='k', linestyle=':', linewidth=2, alpha=0.5, label='Optimal')
    ax4.set_xlabel('Number of Nodes', fontsize=12)
    ax4.set_ylabel('Approximation Ratio', fontsize=12)
    ax4.set_title('Solution Quality: Approximation Ratio (Cut/Optimal)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(f'Models/qaoa_vs_ml/execution_{str(sys.argv[1])}/{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    # === Print Summary Statistics ===
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*70)
    
    for i, num_nodes in enumerate(node_sizes):
        print(f"\n{'='*70}")
        print(f"Node Size: {num_nodes} | Optimal Cut: {optimal_cuts[i]}")
        print(f"{'='*70}")
        
        # Classical results
        classical_ratio = classical_cuts[i] / optimal_cuts[i] if optimal_cuts[i] > 0 else 0
        print(f"\nClassical QAOA:")
        print(f"  Time: {classical_times[i]:.4f}s")
        print(f"  Cut: {classical_cuts[i]}/{optimal_cuts[i]} ({classical_ratio:.2%})")
        
        # Model results
        for model_name in results['model_times'].keys():
            model_time = results['model_times'][model_name][i]
            model_cut = results['model_cuts'][model_name][i]
            model_ratio = model_cut / optimal_cuts[i] if optimal_cuts[i] > 0 else 0
            
            speedup = classical_times[i] / model_time
            quality_diff = model_ratio - classical_ratio
            
            print(f"\n{model_name} Model:")
            print(f"  Time: {model_time:.4f}s (speedup: {speedup:.1f}x)")
            print(f"  Cut: {model_cut}/{optimal_cuts[i]} ({model_ratio:.2%})")
            print(f"  Quality vs Classical: {quality_diff:+.2%}")
    
    # Overall averages
    print(f"\n{'='*70}")
    print("OVERALL AVERAGES")
    print(f"{'='*70}")
    
    avg_classical_time = np.mean(classical_times)
    avg_classical_ratio = np.mean([c/o for c, o in zip(classical_cuts, optimal_cuts)])
    
    print(f"\nClassical QAOA:")
    print(f"  Avg Time: {avg_classical_time:.4f}s")
    print(f"  Avg Approximation Ratio: {avg_classical_ratio:.2%}")
    
    for model_name in results['model_times'].keys():
        avg_model_time = np.mean(results['model_times'][model_name])
        avg_model_ratio = np.mean([c/o for c, o in zip(results['model_cuts'][model_name], optimal_cuts)])
        avg_speedup = avg_classical_time / avg_model_time
        
        print(f"\n{model_name} Model:")
        print(f"  Avg Time: {avg_model_time:.4f}s (avg speedup: {avg_speedup:.1f}x)")
        print(f"  Avg Approximation Ratio: {avg_model_ratio:.2%}")
        print(f"  Quality Difference: {(avg_model_ratio - avg_classical_ratio):+.2%}")


def create_comprehensive_comparison_V2(results, save_prefix="comparison"):
    """
    Creates comprehensive comparison graphs for both timing and fitness
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    node_sizes = results['node_sizes']
    classical_times = results['classical_times']
    classical_cuts = results['classical_cuts']
    optimal_cuts = results['optimal_cuts']

    colors = ['blue', 'green', 'purple', 'orange', 'brown']

    # === Plot 1: Time Comparison (Linear Scale) ===
    ax1 = axes[0, 0]
    ax1.plot(node_sizes, classical_times, 'o-', linewidth=2, markersize=8,
             label='Classical QAOA', color='red', alpha=0.8)

    for i, (model_name, times) in enumerate(results['model_times'].items()):
        ax1.plot(node_sizes, times, 's--', linewidth=2, markersize=8,
                 label=f'{model_name}', color=colors[i % len(colors)], alpha=0.8)

    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === Plot 2: Time Comparison (Log Scale) ===
    ax2 = axes[0, 1]
    ax2.plot(node_sizes, classical_times, 'o-', linewidth=2, markersize=8,
             label='Classical QAOA', color='red', alpha=0.8)

    for i, (model_name, times) in enumerate(results['model_times'].items()):
        ax2.plot(node_sizes, times, 's--', linewidth=2, markersize=8,
                 label=f'{model_name}', color=colors[i % len(colors)], alpha=0.8)

    ax2.set_xlabel('Number of Nodes', fontsize=12)
    ax2.set_ylabel('Time (seconds, log scale)', fontsize=12)
    ax2.set_title('Execution Time Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # === Plot 3: Cut Quality (Absolute Values) ===
    ax3 = axes[1, 0]
    ax3.plot(node_sizes, optimal_cuts, 'k*-', linewidth=3, markersize=12,
             label='Optimal/Best Known', zorder=10, alpha=0.9)
    ax3.plot(node_sizes, classical_cuts, 'o-', linewidth=2, markersize=8,
             label='Classical QAOA', color='red', alpha=0.8)

    for i, (model_name, cuts) in enumerate(results['model_cuts'].items()):
        ax3.plot(node_sizes, cuts, 's--', linewidth=2, markersize=8,
                 label=f'{model_name}', color=colors[i % len(colors)], alpha=0.8)

    ax3.set_xlabel('Number of Nodes', fontsize=12)
    ax3.set_ylabel('Cut Value', fontsize=12)
    ax3.set_title('Solution Quality: Max-Cut Values', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # === Plot 4: Approximation Ratio ===
    ax4 = axes[1, 1]

    # Calculate approximation ratios
    classical_ratios = [c / o if o > 0 else 0 for c, o in zip(classical_cuts, optimal_cuts)]
    ax4.plot(node_sizes, classical_ratios, 'o-', linewidth=2, markersize=8,
             label='Classical QAOA', color='red', alpha=0.8)

    for i, (model_name, cuts) in enumerate(results['model_cuts'].items()):
        model_ratios = [c / o if o > 0 else 0 for c, o in zip(cuts, optimal_cuts)]
        ax4.plot(node_sizes, model_ratios, 's--', linewidth=2, markersize=8,
                 label=f'{model_name}', color=colors[i % len(colors)], alpha=0.8)

    ax4.axhline(y=1.0, color='k', linestyle=':', linewidth=2, alpha=0.5, label='Optimal')
    ax4.set_xlabel('Number of Nodes', fontsize=12)
    ax4.set_ylabel('Approximation Ratio', fontsize=12)
    ax4.set_title('Solution Quality: Approximation Ratio (Cut/Optimal)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(f'Models/qaoa_vs_ml/execution_{str(sys.argv[1])}/{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # === Print Summary Statistics ===
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("=" * 70)
    data_node_size=[]
    data_model=[]
    data_time=[]
    data_time_speedup=[]
    data_cut=[]
    data_cut_optimal=[]
    data_cuts_ratio=[]
    data_vanilla_vs_ml=[]

    for i, num_nodes in enumerate(node_sizes):
        print(f"\n{'=' * 70}")
        print(f"Node Size: {num_nodes} | Optimal Cut: {optimal_cuts[i]}")
        print(f"{'=' * 70}")

        # Classical results
        classical_ratio = classical_cuts[i] / optimal_cuts[i] if optimal_cuts[i] > 0 else 0
        print(f"\nClassical QAOA:")
        print(f"  Time: {classical_times[i]:.4f}s")
        print(f"  Cut: {classical_cuts[i]}/{optimal_cuts[i]} ({classical_ratio:.2%})")

        # STore Vanilla QAOA Results
        data_node_size.append(num_nodes)
        data_model.append('Vanilla')
        data_time.append(classical_times[i])
        data_time_speedup.append(0) # No speedup
        data_cut.append(classical_cuts[i])
        data_cut_optimal.append(optimal_cuts[i])
        data_cuts_ratio.append(classical_ratio)
        data_vanilla_vs_ml.append(0) # it is the same as vanilla

        # Model results

        for model_name in results['model_times'].keys():
            model_time = results['model_times'][model_name][i]
            model_cut = results['model_cuts'][model_name][i]
            model_ratio = model_cut / optimal_cuts[i] if optimal_cuts[i] > 0 else 0

            speedup = classical_times[i] / model_time
            quality_diff = model_ratio - classical_ratio

            print(f"\n{model_name} Model:")
            print(f"  Time: {model_time:.4f}s (speedup: {speedup:.1f}x)")
            print(f"  Cut: {model_cut}/{optimal_cuts[i]} ({model_ratio:.2%})")
            print(f"  Quality vs Classical: {quality_diff:+.2%}")

            # ML-based results
            data_node_size.append(num_nodes)
            data_model.append(model_name)
            data_time.append(model_time)
            data_time_speedup.append(speedup)  # No speedup
            data_cut.append(model_cut)
            data_cut_optimal.append(optimal_cuts[i])
            data_cuts_ratio.append(model_ratio)
            data_vanilla_vs_ml.append(quality_diff)

    data = {'Size':data_node_size, 'Approach': data_model, 'Time': data_time, 'Time_Speed_Up': data_time_speedup, 'Cut': data_cut, 'Optimal_Cut': data_cut_optimal,'Cut_Ratio': data_cuts_ratio, 'Tradeoff':  data_vanilla_vs_ml}
    results_formatted = pd.DataFrame(data)
    results_formatted.to_csv('Models/qaoa_vs_ml/execution_'+ str(sys.argv[1])+'/vanila_vs_ml.csv')

    # Overall averages
    print(f"\n{'=' * 70}")
    print("OVERALL AVERAGES")
    print(f"{'=' * 70}")

    avg_classical_time = np.mean(classical_times)
    avg_classical_ratio = np.mean([c / o for c, o in zip(classical_cuts, optimal_cuts)])

    print(f"\nClassical QAOA:")
    print(f"  Avg Time: {avg_classical_time:.4f}s")
    print(f"  Avg Approximation Ratio: {avg_classical_ratio:.2%}")

    for model_name in results['model_times'].keys():
        avg_model_time = np.mean(results['model_times'][model_name])
        avg_model_ratio = np.mean([c / o for c, o in zip(results['model_cuts'][model_name], optimal_cuts)])
        avg_speedup = avg_classical_time / avg_model_time

        print(f"\n{model_name} Model:")
        print(f"  Avg Time: {avg_model_time:.4f}s (avg speedup: {avg_speedup:.1f}x)")
        print(f"  Avg Approximation Ratio: {avg_model_ratio:.2%}")
        print(f"  Quality Difference: {(avg_model_ratio - avg_classical_ratio):+.2%}")
        
        
def create_detailed_timing_analysis(timing_results):
    """
    Creates additional analysis plots and statistics
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    node_sizes = timing_results['node_sizes']
    classical_times = timing_results['classical_times']
    
    # Plot 1: Linear scale comparison
    ax1.plot(node_sizes, classical_times, 'o-', label='Classical QAOA', linewidth=2)
    for model_name, times in timing_results['model_times'].items():
        ax1.plot(node_sizes, times, 's--', label=f'{model_name} Model', linewidth=2)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Test Time (seconds)')
    ax1.set_title('Linear Scale Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup factors
    for model_name, times in timing_results['model_times'].items():
        speedups = [classical_times[i] / times[i] for i in range(len(node_sizes))]
        ax2.plot(node_sizes, speedups, 'o-', label=f'{model_name} vs Classical', linewidth=2)
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speed Improvement Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
    
    # Plot 3: Bar chart for latest node size
    latest_idx = -1
    methods = ['Classical'] + list(timing_results['model_times'].keys())
    times_latest = [classical_times[latest_idx]] + [times[latest_idx] for times in timing_results['model_times'].values()]
    
    bars = ax3.bar(methods, times_latest, alpha=0.7)
    ax3.set_ylabel('Test Time (seconds)')
    ax3.set_title(f'Test Times for {node_sizes[latest_idx]} Nodes')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times_latest):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                f'{time_val:.4f}s', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Efficiency trend
    for model_name, times in timing_results['model_times'].items():
        efficiency = [classical_times[i] / times[i] for i in range(len(node_sizes))]
        ax4.plot(node_sizes, efficiency, 'o-', label=f'{model_name} Efficiency', linewidth=2)
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Efficiency (Classical Time / Model Time)')
    ax4.set_title('Model Efficiency Trends')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Models/qaoa_vs_ml/execution_'+ str(sys.argv[1])+'/detailed_timing_analysis.png', dpi=300, bbox_inches='tight')
    #plt.show()


def run_timing_comparison():
    """
    Main function to run the timing comparison
    """
    node_sizes = [10, 12, 15, 20, 25]
    edge_probs = [0.5, 0.6, 0.7, 0.8, 0.9]
    model_list = ['xgboost','MLP']
    num_graphs_per_combination = 10
    base_dataset_dir = "datasets"
    
    # Run the analysis
    timing_results = generate_dataset_with_timing_and_fitness(node_sizes, edge_probs, num_graphs_per_combination)
    
    # Create detailed analysis
    create_detailed_timing_analysis(timing_results)
    
    return timing_results

run_timing_comparison()
