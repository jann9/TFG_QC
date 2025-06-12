import pandas as pd
import numpy as np
import os
from qiskit_aer import Aer
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, transpile
from typing import Dict



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
    """Extract depth, gate counts, and other metrics from a circuit decomposed into RX, RZ, and CX gates."""
    # Transpile the circuit to decompose into RX, RZ, CX, while preserving measures and barriers
    basis_gates = ['rx', 'rz', 'cx', 'measure', 'barrier']
    decomposed_circuit = transpile(
        circuit,
        basis_gates=basis_gates,
        optimization_level=0  # Disable optimization to maintain structure
    )
    
    # Initialize counts for desired gates (default to 0 if not present)
    gate_counts = decomposed_circuit.count_ops()
    desired_gates = ['rx', 'rz', 'cx']
    features = {gate: gate_counts.get(gate, 0) for gate in desired_gates}
    
    # Include depth, qubit count, and other gates (e.g., measure, barrier)
    return {
        "depth": decomposed_circuit.depth(),
        "num_qubits": decomposed_circuit.num_qubits,
        **features
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
    
    for num_nodes in node_sizes:
        data = []
        dataset_filename = f"dataset_{num_nodes}_nodes.csv" if num_nodes != "full" else "dataset_full.csv"
        dataset_path = os.path.join(base_dir, dataset_filename)
        df = pd.read_csv(dataset_path)
        q_columns = [col for col in df.columns if col.startswith('Q')]
        print(f"Total rows in CSV: {len(df)}")
        if (num_nodes == "full"):
            num_nodes = 25
        for i, row in df.iterrows():
            q_start = 1
            expected_q_size = num_nodes * (num_nodes + 1) // 2
            q_end = q_start + expected_q_size
            # print(num_nodes,expected_q_size)

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
            data.append(features)
            all_data.append(features)
           
        features_df = pd.DataFrame(data)
        df = pd.concat([
            df.drop(columns=q_columns),  # Remove original Q-columns
            features_df              # Add new features
        ], axis=1)
        
        
        df.to_csv(os.path.join(base_dir, f"dataset_{num_nodes}_nodes_Circuit.csv"), index=False)
        print(f"Saved dataset_{num_nodes}_nodes_Circuit.csv")

    
    
    all_features_df = pd.DataFrame(all_data)
    df = pd.read_csv("datasets/dataset_full.csv")
    df_all = pd.concat([
        df.drop(columns=q_columns),  # Remove original Q-columns
        all_features_df              # Add new features
    ], axis=1)
    df_all.to_csv(os.path.join(base_dir, "dataset_full_Circuit.csv"), index=False)
    print("Saved dataset_full_Circuit.csv")
    

# Example usage
node_sizes = [10, 12, 15, 20, 25]
edge_probs = [0.5, 0.6, 0.7, 0.8, 0.9]
num_graphs_per_combination = 10

generate_dataset(node_sizes)
