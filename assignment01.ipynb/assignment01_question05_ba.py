import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ba generation function
def generate_barabasi_albert_graph(n=5000, target_avg_degree=10, seed=None):
    # calculate m to get target average degree
    # <k> = 2m, so m = <k>/2
    m = int(target_avg_degree / 2)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G, m

# generate 10 independent graph realizations with different random seeds
num_realizations = 10
barabasi_albert_graphs = []

for i in range(num_realizations):
    seed = 42 + i
    G, m = generate_barabasi_albert_graph(n=5000, target_avg_degree=10, seed=seed)
    barabasi_albert_graphs.append(G)

# remove a fraction of nodes uniformly at random from a graph
def remove_nodes_uniformly(G, fraction, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    G_new = G.copy()
    num_nodes_to_remove = int(fraction * G.number_of_nodes())
    
    if num_nodes_to_remove > 0:
        all_nodes = list(G.nodes())
        removed_nodes = np.random.choice(all_nodes, size=num_nodes_to_remove, replace=False)
        G_new.remove_nodes_from(removed_nodes)
        return G_new, list(removed_nodes)
    
    return G_new, []

# remove nodes by degree (targeted attack - highest degree first)
def remove_nodes_by_degree(G, fraction, seed=None):
    G_new = G.copy()
    num_nodes_to_remove = int(fraction * G.number_of_nodes())
    
    if num_nodes_to_remove <= 0:
        return G_new, []
    
    nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    removed_nodes = [node for node, degree in nodes_by_degree[:num_nodes_to_remove]]
    G_new.remove_nodes_from(removed_nodes)
    
    return G_new, removed_nodes

# remove nodes from each graph with multiple fractions
fractions_to_remove = [i * 0.05 for i in range(21)]
perturbed_graphs_dict = {f: [] for f in fractions_to_remove}

for fraction in fractions_to_remove:
    for i, G in enumerate(barabasi_albert_graphs):
        seed = 100 + int(fraction * 100) + i
        G_perturbed, _ = remove_nodes_uniformly(G, fraction, seed=seed)
        perturbed_graphs_dict[fraction].append(G_perturbed)

# compute statistics for each fraction (random failures)
normalized_lcc_sizes = {}
original_N = 5000

for fraction in fractions_to_remove:
    graphs = perturbed_graphs_dict[fraction]
    all_lcc_sizes = []
    
    for G in graphs:
        if G.number_of_nodes() > 0:
            if nx.is_connected(G):
                lcc_size = G.number_of_nodes()
            else:
                connected_components = nx.connected_components(G)
                lcc_size = max(len(c) for c in connected_components)
            all_lcc_sizes.append(lcc_size)
        else:
            all_lcc_sizes.append(0)
    
    avg_lcc_size = np.mean(all_lcc_sizes)
    normalized_lcc_sizes[fraction] = avg_lcc_size / original_N

# targeted attack: remove nodes by degree (highest degree first)
targeted_graphs_dict = {f: [] for f in fractions_to_remove}

for fraction in fractions_to_remove:
    for i, G in enumerate(barabasi_albert_graphs):
        G_targeted, _ = remove_nodes_by_degree(G, fraction)
        targeted_graphs_dict[fraction].append(G_targeted)

# compute statistics for targeted attack
normalized_lcc_sizes_targeted = {}

for fraction in fractions_to_remove:
    graphs = targeted_graphs_dict[fraction]
    all_lcc_sizes = []
    
    for G in graphs:
        if G.number_of_nodes() > 0:
            if nx.is_connected(G):
                lcc_size = G.number_of_nodes()
            else:
                connected_components = nx.connected_components(G)
                lcc_size = max(len(c) for c in connected_components)
            all_lcc_sizes.append(lcc_size)
        else:
            all_lcc_sizes.append(0)
    
    avg_lcc_size = np.mean(all_lcc_sizes)
    normalized_lcc_sizes_targeted[fraction] = avg_lcc_size / original_N

# plot S(f) versus f, comparison of random vs targeted
f_values = sorted(normalized_lcc_sizes.keys())
S_values_random = [normalized_lcc_sizes[f] for f in f_values]
S_values_targeted = [normalized_lcc_sizes_targeted[f] for f in f_values]

plt.figure(figsize=(10, 7))
plt.plot(f_values, S_values_random, 'k-', marker='o', markersize=8, linewidth=2.5, 
         label='Random Failure', alpha=0.8)
plt.plot(f_values, S_values_targeted, color='dimgray', linestyle='--', marker='s', 
         markersize=8, linewidth=2.5, label='Targeted Attack', alpha=0.8)
plt.xlabel('Fraction of nodes removed (f)', fontsize=13)
plt.ylabel('Normalized LCC size S(f) = S/N', fontsize=13)
plt.title('Network Robustness: Random Failure vs Targeted Attack\nBarabási-Albert Graph (N=5000, <k>=10)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(-0.02, max(f_values) + 0.02)
plt.ylim(0, 1.05)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('assignment01_question05_ba_robustness.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved as 'assignment01_question05_ba_robustness.png'")
print("\nAnalysis complete!")