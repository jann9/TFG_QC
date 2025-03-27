import pandas as pd
import numpy as np
import os 
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

def generate_dataset(node_sizes, edge_probs, num_graphs_per_combination, base_dir="datasets"):
    os.makedirs(base_dir, exist_ok=True)

    max_size = max(node_sizes)  # Max node size for zero padding
    all_data = []  # To store data for the unified dataset
    
    for num_nodes in node_sizes:
        data = []
        for prob in edge_probs:
            for graph_index in range(num_graphs_per_combination):
                G = extract_graph(num_nodes=num_nodes, probability=prob, graph_index=graph_index)
                Q, node_mapping = maxcut_qubo(G)
                # Convert QUBO to Pauli Hamiltonian
                paulis = get_pauli_list(Q)
                H = SparsePauliOp.from_list(paulis)
                # QAOA circuit
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
                    x=result.x
                
                # Create column names
                features = extract_upper_triangular(Q)
                features_padded = padding_extract_upper_triangular(Q, max_size)
                row = np.concatenate([[num_nodes, prob], features, x])
                row_padded = np.concatenate([[num_nodes, prob], features_padded, x])
                data.append(row)
                all_data.append(row_padded)
                print(len(features),len(x))       
                print(num_nodes, prob, graph_index, x, "\n")
                
        feature_columns = [f"Q_{i}_{j}" for i in range(num_nodes) for j in range(i, num_nodes)]
        output_columns = [f"x_{i}" for i, val in enumerate(x)]
        columns = ["num_nodes", "edge_prob"] + feature_columns + output_columns
        print(len(feature_columns),len(output_columns))   
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(base_dir, f"dataset_{num_nodes}_nodes.csv"), index=False)
        print(f"Saved dataset_{num_nodes}_nodes.csv")

    
    
    # Save to CSV
    df_all = pd.DataFrame(all_data, columns=columns)
    df_all.to_csv(os.path.join(base_dir, "dataset_full.csv"), index=False)
    print("Saved dataset_full.csv")

# Example usage
node_sizes = [10, 12, 15, 20, 25]
edge_probs = [0.5, 0.6, 0.7, 0.8, 0.9]
num_graphs_per_combination = 10

generate_dataset(node_sizes, edge_probs, num_graphs_per_combination)
