import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 1. Load the Data ---
print("--- [1] Loading Data ---")
try:
    df = pd.read_csv("email-EuAll.txt", sep='\t', comment='#', names=['FromNodeId', 'ToNodeId'])
    G = nx.from_pandas_edgelist(df, 'FromNodeId', 'ToNodeId', create_using=nx.DiGraph())
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
except FileNotFoundError:
    print("Error: 'email-EuAll.txt' not found.")
    exit()

# --- 2. Top 10 Senders vs Top 10 Receivers ---
print("\n--- [2] Analyzing Top Senders and Receivers ---")

# Get Top 10 Senders (Out-Degree)
out_degree_dict = dict(G.out_degree())
top_10_senders = sorted(out_degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]

# Get Top 10 Receivers (In-Degree)
in_degree_dict = dict(G.in_degree())
top_10_receivers = sorted(in_degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]

# Plotting Comparison
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Senders Plot
s_ids = [str(x[0]) for x in top_10_senders]
s_vals = [x[1] for x in top_10_senders]
ax[0].bar(s_ids, s_vals, color='teal', edgecolor='black')
ax[0].set_title('Top 10 Senders (Out-Degree)')
ax[0].set_ylabel('Number of Emails Sent')

# Receivers Plot
r_ids = [str(x[0]) for x in top_10_receivers]
r_vals = [x[1] for x in top_10_receivers]
ax[1].bar(r_ids, r_vals, color='salmon', edgecolor='black')
ax[1].set_title('Top 10 Receivers (In-Degree)')
ax[1].set_ylabel('Number of Emails Received')

plt.tight_layout()
plt.savefig('senders_vs_receivers.png')
print("Saved 'senders_vs_receivers.png'")

# --- 3. PageRank (Influence Analysis) ---
print("\n--- [3] Calculating PageRank ---")
pagerank = nx.pagerank(G, alpha=0.85)
top_10_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]

# Plot PageRank
pr_ids = [str(x[0]) for x in top_10_pr]
pr_scores = [x[1] for x in top_10_pr]

plt.figure(figsize=(10, 6))
plt.bar(pr_ids, pr_scores, color='gold', edgecolor='black')
plt.title('Top 10 Influential Nodes (PageRank Score)')
plt.xlabel('Node ID')
plt.ylabel('PageRank Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('pagerank_scores.png')
print("Saved 'pagerank_scores.png'")

print("Top 10 Influencers by PageRank:")
for node, score in top_10_pr:
    print(f"  Node {node}: Score {score:.5f}")

# --- 4. Community Detection (Explanation below) ---
print("\n--- [4] Detecting Communities ---")
G_undirected = G.to_undirected()
communities = list(nx.community.label_propagation_communities(G_undirected))

# Sort communities by size
communities = sorted(communities, key=len, reverse=True)
print(f"Detected {len(communities)} distinct communities.")
print(f"Largest community size: {len(communities[0])} nodes")

# Plot Community Size Distribution (Top 10)
comm_sizes = [len(c) for c in communities[:10]]
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), comm_sizes, color='mediumpurple', edgecolor='black')
plt.title('Top 10 Largest Communities')
plt.xlabel('Community Rank')
plt.ylabel('Number of Members')
plt.savefig('communities.png')
print("Saved 'communities.png'")

# --- 5. Resilience Test ---
print("\n--- [5] Simulation: Removing Top Sender ---")
hub_node = top_10_senders[0][0]
initial_scc = len(max(nx.strongly_connected_components(G), key=len))

G_attacked = G.copy()
G_attacked.remove_node(hub_node)
new_scc = len(max(nx.strongly_connected_components(G_attacked), key=len))

print(f"Removed Top Sender (Node {hub_node})")
print(f"Initial Core Size: {initial_scc}")
print(f"New Core Size: {new_scc}")
print(f"Change: {initial_scc - new_scc} nodes")

print("\nAnalysis Complete.")