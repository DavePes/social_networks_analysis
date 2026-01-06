# f = open("email-EuAll.txt", "r")
# lines = f.readlines()
# f.close()
# for line in lines[5:]:
#     print(line)
#     x = 3
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 1. Load the Data ---
print("Loading data... (This might take a moment)")
# We use pandas to read the file first as it's faster than the native networkx reader
# We skip the first 4 rows which are comments (starting with #)
try:
    df = pd.read_csv("email-EuAll.txt", sep='\t', comment='#', names=['FromNodeId', 'ToNodeId'])
    # Create a Directed Graph
    G = nx.from_pandas_edgelist(df, 'FromNodeId', 'ToNodeId', create_using=nx.DiGraph())
    print(f"Graph loaded successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
except FileNotFoundError:
    print("Error: File 'email-EuAll.txt' not found. Please make sure it is in the same directory.")
    exit()

# --- 2. Top 10 Biggest Nodes & Connection Percentage ---
print("Analyzing Top 10 Nodes...")
# "Biggest" here is defined by Total Degree (In + Out)
degree_dict = dict(G.degree())
# Sort nodes by degree descending
sorted_degree = sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)
top_10_nodes = sorted_degree[:10]

# Calculate percentage of total graph edges connected to each node
total_edges = G.number_of_edges()
top_nodes_ids = [str(n) for n, d in top_10_nodes]
top_nodes_pct = [(d / total_edges) * 100 for n, d in top_10_nodes]

# Plot Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(top_nodes_ids, top_nodes_pct, color='skyblue', edgecolor='black')
plt.xlabel('Node ID')
plt.ylabel('Percentage of Total Network Edges (%)')
plt.title('Top 10 Nodes by Connectivity (Degree)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom')
plt.savefig('top_10_nodes.png')
print("Saved 'top_10_nodes.png'")

# --- 3. Diameter of Largest Component ---
# "Minimum number of connections to get from one node to all others" = Diameter
print("Calculating Diameter (this may be slow)...")

# We must look at the Strongly Connected Component (SCC) to ensure reachability
if nx.is_strongly_connected(G):
    largest_scc = G
else:
    # Find the largest SCC
    scc_components = list(nx.strongly_connected_components(G))
    largest_scc_nodes = max(scc_components, key=len)
    largest_scc = G.subgraph(largest_scc_nodes)

print(f"Largest SCC has {largest_scc.number_of_nodes()} nodes.")
# Note: Exact diameter on 34k nodes (SCC size) can be very slow (O(NM)). 
# If it hangs, use approximation: nx.approximation.diameter(largest_scc)
try:
    scc_diameter = nx.diameter(largest_scc)
    print(f"Diameter of Largest SCC: {scc_diameter}")
except Exception as e:
    print(f"Could not calculate exact diameter: {e}")

# --- 4. Clustering (Community Detection) ---
print("Analyzing Communities...")
# We use the largest Weakly Connected Component for community detection
# and convert to undirected for simpler modularity analysis.
wcc_components = list(nx.weakly_connected_components(G))
largest_wcc_nodes = max(wcc_components, key=len)
largest_wcc = G.subgraph(largest_wcc_nodes).to_undirected()

# Using Greedy Modularity algorithm
# Note: On 200k nodes this might take a few minutes. 
communities = nx.community.greedy_modularity_communities(largest_wcc)
print(f"Found {len(communities)} communities.")

# Plot Size of Top 10 Communities
comm_sizes = [len(c) for c in communities]
comm_sizes.sort(reverse=True)

plt.figure(figsize=(8, 5))
plt.bar(range(1, min(11, len(comm_sizes)+1)), comm_sizes[:10], color='lightgreen', edgecolor='black')
plt.xlabel('Community Rank (by size)')
plt.ylabel('Number of Nodes in Community')
plt.title('Size of Top 10 Detected Communities')
plt.savefig('communities_size.png')
print("Saved 'communities_size.png'")

# --- 5. Degree Distributions (In vs Out) ---
print("Plotting Degree Distributions...")
in_degrees = [d for n, d in G.in_degree()]
out_degrees = [d for n, d in G.out_degree()]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# In-Degree
ax[0].hist(in_degrees, bins=50, color='salmon', edgecolor='black', log=True)
ax[0].set_title('In-Degree Distribution (Log Scale)')
ax[0].set_xlabel('In-Degree (Incoming Emails)')
ax[0].set_ylabel('Frequency (Number of Nodes)')

# Out-Degree
ax[1].hist(out_degrees, bins=50, color='teal', edgecolor='black', log=True)
ax[1].set_title('Out-Degree Distribution (Log Scale)')
ax[1].set_xlabel('Out-Degree (Outgoing Emails)')
ax[1].set_ylabel('Frequency')

plt.savefig('degree_distribution.png')
print("Saved 'degree_distribution.png'")
print("Analysis Complete.")