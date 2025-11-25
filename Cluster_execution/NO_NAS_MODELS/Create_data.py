import networkx as nx
import numpy as np
import json
import pandas as pd
import os

def generate_maxcut_graph(n, p, weighted=True):
    """Generate an Erdős–Rényi random graph with edge weights."""
    G = nx.erdos_renyi_graph(n, p)
    
    if weighted:
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.1, 1.0)  # Random edge weights

    return G

# Define parameters
node_sizes = [10, 12, 15, 20, 25]
edge_probs = [0.5, 0.6, 0.7, 0.8, 0.9]
num_graphs_per_combination = 10  # Total graphs = 5 * 5 * 10 = 250

# Prepare folders for different formats
os.makedirs("datasets", exist_ok=True)

# JSON storage
json_data = []

# CSV storage (edge lists)
csv_data = []

# NumPy storage (adjacency matrices)
adj_matrices = {}

# Generate dataset
for n in node_sizes:
    for p in edge_probs:
        for idx in range(num_graphs_per_combination):
            G = generate_maxcut_graph(n, p, weighted=True)
            edges = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
            
            # JSON format
            json_data.append({"num_nodes": n, "probability": p, "edges": edges})

            # CSV format
            for u, v, w in edges:
                csv_data.append([n, p, idx, u, v, w])  # Store node size, p, graph index, edge info
            
            # NumPy format (Adjacency matrix)
            A = nx.to_numpy_array(G, weight='weight')
            key = f"graph_n{n}_p{p}_idx{idx}"
            adj_matrices[key] = A  # Store adjacency matrix separately

# Save JSON
json_filename = "datasets/maxcut_dataset.json"
with open(json_filename, "w") as f:
    json.dump(json_data, f, indent=4)

# Save CSV
csv_filename = "datasets/maxcut_dataset.csv"
df = pd.DataFrame(csv_data, columns=["num_nodes", "probability", "graph_index", "node_1", "node_2", "weight"])
df.to_csv(csv_filename, index=False)

# Save NumPy (.npz) correctly
npz_filename = "datasets/maxcut_dataset.npz"
np.savez_compressed(npz_filename, **adj_matrices)

print(f"Dataset saved in multiple formats:\n- JSON: {json_filename}\n- CSV: {csv_filename}\n- NumPy: {npz_filename}")





# Load CSV dataset
csv_filename = "datasets/maxcut_dataset.csv"
df = pd.read_csv(csv_filename)

def extract_graph(df, num_nodes, probability, graph_index):
    """Extracts a specific graph from the dataset and returns a NetworkX graph."""
    sub_df = df[(df["num_nodes"] == num_nodes) & 
                (df["probability"] == probability) & 
                (df["graph_index"] == graph_index)]
    
    G = nx.Graph()
    for _, row in sub_df.iterrows():
        G.add_edge(int(row["node_1"]), int(row["node_2"]), weight=row["weight"])
    
    return G

# Example: Extract a graph with 10 nodes, p=0.5, index=0
G = extract_graph(df, num_nodes=10, probability=0.5, graph_index=0)

def maxcut_qubo(G):
    """Generate QUBO matrix for MaxCut problem, mapping nodes to indices."""
    node_list = list(G.nodes())  # Get list of nodes
    node_index = {node: idx for idx, node in enumerate(node_list)}  # Map nodes to indices
    n = len(node_list)  # Number of nodes

    Q = np.zeros((n, n))  # Initialize QUBO matrix

    for i, j in G.edges():
        w = G[i][j]['weight']
        idx_i = node_index[i]  # Convert node to index
        idx_j = node_index[j]  # Convert node to index
        Q[idx_i, idx_i] -= w
        Q[idx_j, idx_j] -= w
        Q[idx_i, idx_j] += 2 * w  # Off-diagonal term

    return Q, node_list  # Return QUBO and node mapping

Q, node_mapping = maxcut_qubo(G)

print("QUBO Matrix for QAOA:\n", Q)
print("Node Mapping:", node_mapping)
