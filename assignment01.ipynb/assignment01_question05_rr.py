import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#this code performs robustness analysis on random regular graphs
# rr generation function
def generate_random_regular_graph(n=5000, d=10, seed=None):
    if (n * d) % 2 != 0:
        raise ValueError(f"N*d must be even. Got N={n}, d={d}")
    
    G = nx.random_regular_graph(d, n, seed=seed)
    return G, d

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

def remove_nodes_by_degree(G, fraction, seed=None):
    G_new = G.copy()
    num_nodes_to_remove = int(fraction * G.number_of_nodes())
    
    if num_nodes_to_remove <= 0:
        return G_new, []
    
    nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    removed_nodes = [node for node, degree in nodes_by_degree[:num_nodes_to_remove]]
    G_new.remove_nodes_from(removed_nodes)
    
    return G_new, removed_nodes

def compute_lcc_sizes(graphs_dict, original_N):
    normalized_lcc_sizes = {}
    
    for fraction, graphs in graphs_dict.items():
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
    
    return normalized_lcc_sizes

# generate 10 independent graph realizations
print("Generating 10 Random Regular graphs (N=5000, d=10)")
num_realizations = 10
random_regular_graphs = []

for i in range(num_realizations):
    seed = 42 + i
    G, d = generate_random_regular_graph(n=5000, d=10, seed=seed)
    random_regular_graphs.append(G)

print(f"Generated {len(random_regular_graphs)} graphs")

# define fractions to test
fractions_to_remove = [i * 0.05 for i in range(21)]
original_N = 5000

# random failure analysis
print("\nPerforming random failure analysis")
perturbed_graphs_dict = {f: [] for f in fractions_to_remove}

for fraction in fractions_to_remove:
    for i, G in enumerate(random_regular_graphs):
        seed = 100 + int(fraction * 100) + i
        G_perturbed, _ = remove_nodes_uniformly(G, fraction, seed=seed)
        perturbed_graphs_dict[fraction].append(G_perturbed)

normalized_lcc_sizes_random = compute_lcc_sizes(perturbed_graphs_dict, original_N)

# targeted attack analysis
print("Performing targeted attack analysis")
targeted_graphs_dict = {f: [] for f in fractions_to_remove}

for fraction in fractions_to_remove:
    for i, G in enumerate(random_regular_graphs):
        G_targeted, _ = remove_nodes_by_degree(G, fraction)
        targeted_graphs_dict[fraction].append(G_targeted)

normalized_lcc_sizes_targeted = compute_lcc_sizes(targeted_graphs_dict, original_N)

# plot comparison
print("Creating plot")
f_values = sorted(normalized_lcc_sizes_random.keys())
S_values_random = [normalized_lcc_sizes_random[f] for f in f_values]
S_values_targeted = [normalized_lcc_sizes_targeted[f] for f in f_values]

plt.figure(figsize=(10, 7))
plt.plot(f_values, S_values_random, 'k-', marker='o', markersize=8, linewidth=2.5, 
         label='Random Failure', alpha=0.8)
plt.plot(f_values, S_values_targeted, color='dimgray', linestyle='--', marker='s', 
         markersize=8, linewidth=2.5, label='Targeted Attack', alpha=0.8)
plt.xlabel('Fraction of nodes removed (f)', fontsize=13)
plt.ylabel('Normalized LCC size S(f) = S/N', fontsize=13)
plt.title('Network Robustness: Random Failure vs Targeted Attack\nRandom Regular Graph (N=5000, d=10)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(-0.02, max(f_values) + 0.02)
plt.ylim(0, 1.05)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('assignment01_question05_rr_robustness.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved as 'assignment01_question05_rr_robustness.png'")
print("Analysis complete!")