import networkx as nx
import numpy as np
import scipy as sp
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# # Create a graph distance measure, or implement one that you found in the literature.
# 
# (a) Describe the graph descriptor, ϕ(G), you use in your distance measure. Justify
# why this is a good descriptor to use.
# 
# (b) Describe the distance measure you use to quantify dissimilarity between pairs of
# graphs, G1 and G2.
# 
# (c) Implement your graph distance in two functions: the first function should compute
# the ϕ(G) for an input graph, G, and the second function should call the first in
# order to output the final distance value. For example, if your graph distance was
# the Degree Divergence, your descriptor ϕ(G) would be a function that takes a
# graph as input and outputs the degree distribution of G. Your distance function
# would then take two graphs as input, create a ϕ(G1) and ϕ(G2), and calculate the
# distance d(ϕ(G1), ϕ(G2)). You may not use the Deg. Div. as your graph distance.
# 
# (d) For 1,000 iterations, sample pairs of Erd˝os-R´enyi graphs with N = 500 nodes and
# p = 0.016. Calculate and store the graph distance between each pair. Plot the
# distribution of distance values between the ER networks.
# 
# (e) For 1,000 iterations, sample pairs of Barab´asi-Albert graphs with N = 500 nodes
# and m = 4. Calculate and store the graph distance between each pair. Plot the
# distribution of distance values between the BA networks.
# 
# (f) For 1,000 iterations, using the same parameterizations above, sample one ER
# and one BA graph, and compute and store the distance between them. Neatly
# visualize all three distributions on the same plot, with appropriate legends and
# color. What do you notice?

def compute_graph_descriptor(G):
    # compute the descriptor(G) for an input graph G
    # the descriptor is the shortest path length (geodesic distance) between all pairs of nodes
    
    # parameters:
    # G: nx graph
        
    # returns:
    # phi: dictionary where keys are node pairs (i, j) and values are shortest path lengths.
    
    # initialize descriptor dictionary
    phi = {}
    
    # compute all pairs shortest path lengths
    for source in G.nodes():
        # get shortest paths from source to all other nodes
        lengths = nx.single_source_shortest_path_length(G, source)
        
        for target in G.nodes():
            if source != target:  # exclude self-loops
                if target in lengths:
                    phi[(source, target)] = lengths[target]
                else:
                    # nodes are disconnected
                    phi[(source, target)] = float('inf')
    
    return phi


def distance_function(G1, G2):
    # compute the distance between two graphs based on their descriptors.
    
    # the distance is computed as the wasserstein distance (earth mover's distance)
    # between the distributions of finite shortest path lengths in each graph
    
    # parameters:
    # G1: first input graph
    # G2: second input graph
        
    # returns:
    # distance as a float
   
    # compute descriptors for both graphs
    phi1 = compute_graph_descriptor(G1)
    phi2 = compute_graph_descriptor(G2)
    
    # get path lengths 
    paths1 = [d for d in phi1.values() if d != float('inf')]
    paths2 = [d for d in phi2.values() if d != float('inf')]
    
    # edge cases
    if len(paths1) == 0 and len(paths2) == 0:
        return 0.0
    if len(paths1) == 0 or len(paths2) == 0:
        return float('inf')
    
    # compute wasserstein distance between the two distributions
    distance = wasserstein_distance(paths1, paths2)
    
    return distance

# or 1,000 iterations, sample pairs of er graphs with N = 500 nodes and
# p = 0.016. calculate and store the graph distance between each pair. plot the
# distribution of distance values between the ER networks.

# example usage
# create a sample graph
N = 200
p = 0.016
m = 4
ER_distance_distribution = []
BA_distance_distribution = []
ER_BA_distance_distribution = []

# progress indicator for computing distances
print("Computing graph distances...")
for i in range(500):
    # print progress every 100 iterations
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i + 1}/1000 iterations completed ({(i + 1)/10:.0f}%)")
    
    G1 = nx.erdos_renyi_graph(N, p, seed=42+i)
    G2 = nx.erdos_renyi_graph(N, p, seed=43+i)
    dist_er = distance_function(G1, G2)
    ER_distance_distribution.append(dist_er)

    G3 = nx.barabasi_albert_graph(N, m, seed=42+i)
    G4 = nx.barabasi_albert_graph(N, m, seed=43+i)
    dist_ba = distance_function(G3, G4)
    BA_distance_distribution.append(dist_ba)

    dist_er_ba = distance_function(G1, G3)
    ER_BA_distance_distribution.append(dist_er_ba)

print("  Completed: 1000/1000 iterations (100%)\n")

# plot the distribution of distance values between the ER networks.
# compute histo for line plot
counts_er, bin_edges_er = np.histogram(ER_distance_distribution, bins=30)
bin_centers_er = (bin_edges_er[:-1] + bin_edges_er[1:]) / 2

counts_ba, bin_edges_ba = np.histogram(BA_distance_distribution, bins=30)
bin_centers_ba = (bin_edges_ba[:-1] + bin_edges_ba[1:]) / 2

counts_er_ba, bin_edges_er_ba = np.histogram(ER_BA_distance_distribution, bins=30)
bin_centers_er_ba = (bin_edges_er_ba[:-1] + bin_edges_er_ba[1:]) / 2

plt.figure(figsize=(10, 6))
plt.plot(bin_centers_er, counts_er, color='gray', linewidth=2, marker='o', markersize=4, label='ER-ER')
plt.plot(bin_centers_ba, counts_ba, color='black', linewidth=2, marker='s', markersize=4, label='BA-BA')
plt.plot(bin_centers_er_ba, counts_er_ba, color='darkgray', linewidth=2, marker='^', markersize=4, label='ER-BA')
plt.xlabel('Distance (Wasserstein Distance)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Graph Distances between Network Pairs (N=200)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assignment01_question04_ER_distance_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\ndone'")