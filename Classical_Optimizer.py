import pandas as pd
import numpy as np

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
import matplotlib.pyplot as plt
import networkx as nx

# ----------------- Import dataset


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

# Example: Extract a graph
G = extract_graph(num_nodes=10, probability=0.5, graph_index=0)
print("Graph: \n")
print(G)
print("\n")


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

Q, node_mapping = maxcut_qubo(G)

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

# Convert QUBO to Pauli Hamiltonian
paulis = get_pauli_list(Q)
H = SparsePauliOp.from_list(paulis)
print("\n")
# print(paulis)
# print("\n")
print("Cost Function Hamiltonian: \n", H)


# QAOA circuit

QAOAcircuit = QAOAAnsatz(cost_operator=H, reps=2)
QAOAcircuit.measure_all()

# QAOAcircuit.draw(output="mpl").show()
print(QAOAcircuit) # Print the circuit

print(QAOAcircuit.parameters) # Print the parameters

#-------------------------- Optimization of the parameters

backend = Aer.get_backend('qasm_simulator')
print(backend)

# Create pass manager for transpilation
pm = generate_preset_pass_manager(optimization_level=3,
                                    backend=backend)

candidate_circuit = pm.run(QAOAcircuit)

# print(candidate_circuit)
# candidate_circuit.draw('mpl', fold=False, idle_wires=False)

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
with Session(backend=backend) as session:
    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 1000

    # Set simple error suppression/mitigation options
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, H, estimator),
        method="COBYLA",
        tol=1e-2,
    )
    print(result)




plt.figure(figsize=(12, 6))
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()





















'''

def build_max_cut_paulis(G):
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(G.edges()):
        paulis = ["I"] * len(list(G.nodes()))
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = G[edge[0]][edge[1]]['weight']
        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list

# Print the Hamiltonian
max_cut_paulis = build_max_cut_paulis(G)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
# print(max_cut_paulis)
print("Cost Function Hamiltonian: \n", cost_hamiltonian)
'''
