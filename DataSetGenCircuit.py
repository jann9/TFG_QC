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
from qiskit import QuantumCircuit
from typing import Dict
from qiskit_ibm_runtime import SamplerV2 as Sampler



base_dir = "datasets"


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

def get_circuit_features(circuit: QuantumCircuit) -> Dict[str, float]:
    """Extract depth, gate counts, and other metrics from a transpiled circuit."""
    return {
        "depth": circuit.depth(),
        "num_qubits": circuit.num_qubits,
        **circuit.count_ops(),  # Adds gate counts (e.g., 'cx': 10, 'rz': 5)
    }
    
#-------------------------- Optimization of the parameters

backend = Aer.get_backend('qasm_simulator')
# print(backend)

# Create pass manager for transpilation
pm = generate_preset_pass_manager(optimization_level=3,
                                    backend=backend)

def generate_dataset(node_sizes, base_dir="datasets"):
    os.makedirs(base_dir, exist_ok=True)

    all_data = []  # To store data for the unified dataset
    
    for num_nodes in node_sizes:#+ ["full"]:
        data = []
        dataset_filename = f"dataset_{num_nodes}_nodes.csv" if num_nodes != "full" else "dataset_full.csv"
        dataset_path = os.path.join(base_dir, dataset_filename)
        df = pd.read_csv(dataset_path)
        
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
            
            H = SparsePauliOp.from_list(get_pauli_list(q_matrix))
            # QAOA circuit
            QAOAcircuit = QAOAAnsatz(cost_operator=H, reps=2)
            QAOAcircuit.measure_all()
            QAOAcircuit.assign_parameters(row.filter(like="x_"))
            final_circuit = pm.run(QAOAcircuit)
            features = get_circuit_features(final_circuit)
            for key, value in features.items():
                df.loc[i, key] = value
            '''   
                # Bind optimized parameters to the transpiled circuit
                optimized_circuit = candidate_circuit.assign_parameters(result.x)
                # Create column names
                print(optimized_circuit)
                features = [value for value in optimized_circuit.count_ops().values()] + [optimized_circuit.num_qubits,optimized_circuit.depth()]
                print(optimized_circuit.count_ops())
                row = np.concatenate([[num_nodes, prob], features, x])
                data.append(row)
                all_data.append(row)
                print(len(features),len(x))       
                print(num_nodes, prob, graph_index, x, "\n")
            
        feature_columns = [f"{key}" for key in optimized_circuit.count_ops()]+ ["num_qbits", "depth"]
        output_columns = [f"x_{i}" for i, val in enumerate(x)]
        columns = ["num_nodes", "edge_prob"] + feature_columns + output_columns
        print(len(feature_columns),len(output_columns))   
        df = pd.DataFrame(data, columns=columns)
        '''
        
        df.to_csv(os.path.join(base_dir, f"dataset_{num_nodes}_nodes_Circuit.csv"), index=False)
        print(f"Saved dataset_{num_nodes}_nodes_Circuit.csv")

    
    '''
    # Save to CSV
    df_all = pd.DataFrame(all_data, columns=columns)
    df_all.to_csv(os.path.join(base_dir, "dataset_full_Circuit.csv"), index=False)
    print("Saved dataset_full_Circuit.csv")
    '''

# Example usage
node_sizes = [10]
edge_probs = [0.5]
num_graphs_per_combination = 10

generate_dataset(node_sizes)
