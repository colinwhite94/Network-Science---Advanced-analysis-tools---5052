import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# readme: Network Generation and Growth 
# this code implements the optimization model
# where nodes placed in a 2D plane with basins of attraction are calculated
# new nodes attach to existing nodes based on minimizing a cost function
# the cost function balances 2D distance and network distance to a central node
# the code generates networks with varying delta values to see how it affects structure
# and visualizes the networks and their degree distributions.

def optimization_growth_model(n, m=1, delta=1.0, seed=None):
    # optimization growth model.
    # new nodes attach to the node whose basin they fall into based on minimizing a cost function
    # the cost function balances 2d distance and network distance to a central node
    
    # parameters:
    # n: final number of nodes
    # m:number of links each new node makes (default: 1)
    # delta: weight parameter for the cost function (default: 1.0), balances dial for 2d distance and network distance
    # seed: random seed for reproducibility
        
    # returns:
    # G: networkx.Graph
    # positions: dictionary mapping node ids to their positions in 2d


    if seed is not None:
        np.random.seed(seed)
    
    # initialize graph and position dictionary
    G = nx.Graph()
    positions = {}
    
    # start with a small ba graph as initial network
    m0 = max(m + 3, 5)  # ensure we have enough initial nodes for BA graph
    initial_m = max(m, 2)  # parameter for initial ba graph
    
    # generate initial small ba graph
    G_initial = nx.barabasi_albert_graph(m0, initial_m, seed=seed)
    
    # add nodes with random positions
    for i in G_initial.nodes():
        G.add_node(i)
        positions[i] = np.random.rand(2)
    
    # add all edges from initial ba graph
    for u, v in G_initial.edges():
        G.add_edge(u, v)
    
    # add remaining nodes
    for i in range(m0, n):
        # new node position in 2d
        new_pos = np.random.rand(2)
        positions[i] = new_pos
        
        # calculate 2d distances from new node to all existing nodes
        euclidean_distances = {}
        for j in G.nodes():
            dist = np.linalg.norm(new_pos - positions[j])
            euclidean_distances[j] = dist
        
        # choose central node (node with highest degree, or first node if tie)
        central_node = max(G.nodes(), key=lambda node: (G.degree(node), -node))
        
        # calculate network distances for all existing nodes
        # network distance for node j = shortest path length from j to central node
        network_distances = {}
        central_path_lengths = nx.single_source_shortest_path_length(G, central_node)
        
        # fill network distances
        # for all nodes in G
        # assign network distance based on shortest path to central node
        for j in G.nodes():
            if j in central_path_lengths:
                network_distances[j] = central_path_lengths[j]
            else:
                # if j is not connected to central node (shouldn't happen)
                network_distances[j] = float('inf')
        
        # determine which basin the new node falls into by minimizing cost function
        # cost function is cost(j) = delta * euclidean_distance(new_node, j) + network_distance(j)
        costs = {}
        for j in G.nodes():
            costs[j] = delta * euclidean_distances[j] + network_distances[j]
        
        # select m nodes with minimum cost
        # sort nodes by cost and select the m nodes with lowest cost
        sorted_nodes = sorted(costs.items(), key=lambda x: x[1])
        targets = [node for node, cost in sorted_nodes[:min(m, len(G.nodes()))]]
        
        # add node and edges
        G.add_node(i)
        for target in targets:
            G.add_edge(i, target)
    
    return G, positions

# visualization function
# plot network and degree distribution
# node sizes proportional to degree
def plot_optimization_network(G, positions, title="Optimization Model Network"):    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # plot 1: network in 2d layout
    node_sizes = [100 * (G.degree(node) + 1) for node in G.nodes()]
    node_colors = [G.degree(node) for node in G.nodes()]
    
    # plot network
    nx.draw_networkx_nodes(G, positions, node_size=node_sizes, 
                          node_color=node_colors, cmap='viridis',
                          alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, positions, alpha=0.3, ax=ax1)
    
    # title and labels
    ax1.set_title(f"{title}\n(Node size ∝ degree)", fontsize=14)
    ax1.set_xlabel("x position")
    ax1.set_ylabel("y position")
    ax1.set_aspect('equal')
    
    # plot 2: degree distribution
    degrees = [G.degree(node) for node in G.nodes()]
    degree_counts = {}
    for k in degrees:
        degree_counts[k] = degree_counts.get(k, 0) + 1
    ks = sorted(degree_counts.keys())
    counts = [degree_counts[k] for k in ks]
    
    # plot degree distribution
    # log-log scale
    ax2.loglog(ks, counts, 'bo-', markersize=8, alpha=0.7)
    ax2.set_xlabel("Degree k", fontsize=12)
    ax2.set_ylabel("Count P(k)", fontsize=12)
    ax2.set_title("Degree Distribution", fontsize=14)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# generate networks with different delta values
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# delta values to test
# small delta: network distance dominates
# large delta: distance dominates
deltas = [0.1, 10, 1000]
n_nodes = 500

# generate and plot networks for each delta
# loop through delta values
# create network and plots
for idx, delta in enumerate(deltas):
    # generate network
    G, pos = optimization_growth_model(n=n_nodes, m=2, delta=delta, seed=42)
    
    # vzation
    ax = axes[0, idx]
    spring_pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    node_sizes = [10 * (G.degree(node) + 1) for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, spring_pos, node_size=node_sizes, 
                          node_color='dimgray',
                          alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, spring_pos, alpha=0.15, edge_color='lightgray', ax=ax, width=0.5)
    
    ax.set_title(f"Network Visualization\nδ = {delta}", fontsize=12)
    ax.axis('off')
    
    # degree distribution plot
    ax = axes[1, idx]
    degrees = [G.degree(node) for node in G.nodes()]
    degree_counts = {}
    for k in degrees:
        degree_counts[k] = degree_counts.get(k, 0) + 1
    ks = sorted(degree_counts.keys())
    counts = [degree_counts[k] for k in ks]
    
    # plot degree distribution
    # log-log scale
    ax.loglog(ks, counts, 'k-', marker='o', markersize=6, alpha=0.7)
    ax.set_xlabel("Degree k")
    ax.set_ylabel("Count P(k)")
    ax.set_title(f"Degree Distribution\nδ = {delta}")
    ax.grid(True, alpha=0.3)
    
    # print statistics
    print(f"\nδ = {delta}:")
    print(f"  Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Number of edges: {G.number_of_edges()}")

plt.suptitle("Optimization Model, varying delta", fontsize=16)
plt.tight_layout()
plt.savefig('assignment01_question03_part2.png', dpi=150, bbox_inches='tight')
plt.show()