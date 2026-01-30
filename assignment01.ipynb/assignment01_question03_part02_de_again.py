import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

#Preferential Attachment Measurement for Network Growth Models
#Measures the preferential attachment function Π(k) for four network growth models:
#1: ba: new nodes attach preferentially to high-degree nodes
#2: copying: new nodes copy connections from a prototype (p=0.5)
#3: link-selection: random edge sampling with endpoint selection
#4: optimization: spatial/distance cost minimization (δ=10)

# grows networks from n=5,000 to 10,000 nodes, recording the degree of 
# attached nodes at each step. Outputs a 2×2 plot comparing Π(k) curves.

# generate network growth models

# copying model
# new nodes selects a prototype node at random
# with probability p connects to the prototype
# otherwise connect to a neighbor of the prototype
def copying_model(n, m, p=0.5, seed=None):
    #copying model with probability p for direct connection to prototype
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.Graph()

    # start with m+1 nodes in a complete graph
    for i in range(m + 1):
        G.add_node(i)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            G.add_edge(i, j)

    # add remaining nodes
    for new_node in range(m + 1, n):
        G.add_node(new_node)
        existing_nodes = list(G.nodes())[:-1]
        if not existing_nodes:
            continue
        
        prototype_node = random.choice(existing_nodes)
        prototype_neighbors = list(G.neighbors(prototype_node))

        targets = set()
        
        # if prototype has no neighbors, adjust strategy
        if not prototype_neighbors:
            # all connections must go to prototype (can only add once)
            targets.add(prototype_node)
            # fill remaining with random nodes
            if len(targets) < m:
                available = [node for node in existing_nodes if node not in targets and node != new_node]
                if available:
                    extra = random.sample(available, min(m - len(targets), len(available)))
                    targets.update(extra)
        else:
            # normal copying model logic
            attempts = 0
            max_attempts = max(50 * m, 100)
            
            while len(targets) < m and attempts < max_attempts:
                attempts += 1
                if random.random() < p:
                    candidate = prototype_node
                else:
                    candidate = random.choice(prototype_neighbors)

                if candidate != new_node and candidate not in targets:
                    targets.add(candidate)

            # fill remaining targets if needed
            if len(targets) < m:
                available = [node for node in existing_nodes if node not in targets and node != new_node]
                if available:
                    extra = random.sample(available, min(m - len(targets), len(available)))
                    targets.update(extra)

        for target in list(targets):
            G.add_edge(new_node, target)

    return G

# link-selection model
# new nodes samples edges from existing graph
# randomly select one endpoint of the sampled edge to connect to
# new nodes start with m+1 complete graph
def link_selection_model(n, m, seed=None):
    # link-selection model - samples edges 
    if seed is not None:
        random.seed(seed)
    
    G = nx.Graph()
    
    # start with m+1 nodes in a complete graph
    for i in range(m + 1):
        G.add_node(i)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            G.add_edge(i, j)
    
    # add remaining nodes
    for new_node in range(m + 1, n):
        G.add_node(new_node)
        targets = set()
        edges = list(G.edges())
        
        # sample edges 
        attempts = 0
        max_attempts = m * 100  # Safety limit
        
        while len(targets) < m and attempts < max_attempts and edges:
            attempts += 1
            # sample one edge
            edge = random.choice(edges)
            
            # randomly select one endpoint
            target = random.choice(edge)
            targets.add(target)
        
        # stop if we couldn't get m targets
        if len(targets) < m:
            available_nodes = [node for node in G.nodes() if node != new_node and node not in targets]
            if available_nodes:
                extra_needed = m - len(targets)
                targets.update(random.sample(available_nodes, min(extra_needed, len(available_nodes))))
        
        for target in targets:
            G.add_edge(new_node, target)
    
    return G

# optimization growth model
# new nodes have random 2d positions
# connect to m existing nodes minimizing cost function
# cost(j) = delta * 2d distance(new_node, j) + network_distance(j
def optimization_growth_model(n, m=1, delta=10, seed=None):
    #optimization growth model and cost function
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.Graph()
    positions = {}
    
    # start with initial BA graph
    m0 = max(m + 3, 5)
    initial_m = max(m, 2)
    G_initial = nx.barabasi_albert_graph(m0, initial_m, seed=seed)
    
    for i in G_initial.nodes():
        G.add_node(i)
        positions[i] = np.random.rand(2)
    
    for u, v in G_initial.edges():
        G.add_edge(u, v)
    
    # add remaining nodes
    for i in range(m0, n):
        new_pos = np.random.rand(2)
        positions[i] = new_pos
        
        # calculate 2d   distances
        euclidean_distances = {}
        for j in G.nodes():
            dist = np.linalg.norm(new_pos - positions[j])
            euclidean_distances[j] = dist
        
        # find central node (highest degree)
        central_node = max(G.nodes(), key=lambda node: (G.degree(node), -node))
        
        # calculate network distances from central node
        central_path_lengths = nx.single_source_shortest_path_length(G, central_node)
        network_distances = {}
        for j in G.nodes():
            network_distances[j] = central_path_lengths.get(j, float('inf'))
        
        # calculate costs and select targets
        costs = {}
        for j in G.nodes():
            costs[j] = delta * euclidean_distances[j] + network_distances[j]
        
        sorted_nodes = sorted(costs.items(), key=lambda x: x[1])
        targets = [node for node, cost in sorted_nodes[:min(m, len(G.nodes()))]]
        
        G.add_node(i)
        for target in targets:
            G.add_edge(i, target)
    
    return G, positions

# measurement function
# measures Π(k) by growing the network and recording attachment degrees
def measure_preferential_attachment(n, m, model_name, model_params, num_trials=10000, seed=None, progress=False):
    #Measure Π(k) by growing the network and recording attachment degrees.
    #parameters:
    #n: initial network size
    # m: edges per new node
    # model_name: which model ei 'ba', 'copying', 'link_selection', or 'optimization'
    # model_params: additional parameters for the model
    # num_trials: number of new nodes to add for measurement
    # seed: random seed
 
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # generate initial network
    if model_name == 'ba':
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        positions = None
        # initialize repeated_nodes list for ba preferential attachment
        repeated_nodes = [node for node, deg in G.degree() for _ in range(deg)]
        
    # generate models
    elif model_name == 'copying':
        p = model_params.get('p', 0.5)
        G = copying_model(n, m, p=p, seed=seed)
        positions = None
        repeated_nodes = None
        
    elif model_name == 'link_selection':
        G = link_selection_model(n, m, seed=seed)
        positions = None
        repeated_nodes = None
        
    elif model_name == 'optimization':
        delta = model_params.get('delta', 10)
        G, positions = optimization_growth_model(n, m, delta=delta, seed=seed)
        repeated_nodes = None
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # for recording attachment degrees
    degree_counts = defaultdict(int)  # count of attachments to each degree
    degree_node_counts = defaultdict(int)  # count of nodes with each degree
    total_attachments = 0
    
    # add new nodes and record attachment degrees
    for trial in range(num_trials):
        new_node_id = max(G.nodes()) + 1
        existing_nodes = list(G.nodes())
        
        # determine targets using the actual model mechanism
        if model_name == 'ba':
            # preferential attachment using updated repeated_nodes
            targets = set()
            attempts = 0
            max_attempts = m * 100
            while len(targets) < m and attempts < max_attempts and repeated_nodes:
                attempts += 1
                target = random.choice(repeated_nodes)
                targets.add(target)
            
            # stop if we couldn't get m targets
            if len(targets) < m:
                available = [node for node in existing_nodes if node not in targets]
                if available:
                    extra = random.sample(available, min(m - len(targets), len(available)))
                    targets.update(extra)
            
            targets = list(targets)
            
            # validate
            if len(targets) < m:
                print(f"stop: ba model only got {len(targets)} targets instead of {m}")
    
        elif model_name == 'copying':
            # copying model with probability p
            p = model_params.get('p', 0.5)
            prototype_node = random.choice(existing_nodes)
            prototype_neighbors = list(G.neighbors(prototype_node))
            
            targets = set()
            
            # if prototype has no neighbors, adjust strategy
            if not prototype_neighbors:
                # all connections must go to prototype (can only add once)
                targets.add(prototype_node)
                # fill remaining with random nodes
                if len(targets) < m:
                    available = [node for node in existing_nodes if node not in targets]
                    if available:
                        extra = random.sample(available, min(m - len(targets), len(available)))
                        targets.update(extra)
            else:
                # normal copying model logic
                attempts = 0
                max_attempts = max(50 * m, 100)
                
                while len(targets) < m and attempts < max_attempts:
                    attempts += 1
                    if random.random() < p:
                        candidate = prototype_node
                    else:
                        candidate = random.choice(prototype_neighbors)
                    
                    if candidate not in targets:
                        targets.add(candidate)
                
                # fill remaining if needed
                if len(targets) < m:
                    available = [node for node in existing_nodes if node not in targets]
                    if available:
                        extra = random.sample(available, min(m - len(targets), len(available)))
                        targets.update(extra)
            
            targets = list(targets)
            
            # validate
            if len(targets) < m:
                print(f"stop: copying model only got {len(targets)} targets instead of {m}")
            
        elif model_name == 'link_selection':
            # link-selection model, sample edges 
            edges = list(G.edges())
            targets = set()
            
            # sample edges with replacement until we have m unique targets
            attempts = 0
            max_attempts = m * 100  # safety limit
            
            while len(targets) < m and attempts < max_attempts and edges:
                attempts += 1
                # sample one edge (with replacement)
                edge = random.choice(edges)
                
                # randomly select one endpoint
                target = random.choice(edge)
                targets.add(target)
            
            # fallback if we couldn't get m targets
            if len(targets) < m:
                available_nodes = [node for node in existing_nodes if node not in targets]
                if available_nodes:
                    extra_needed = m - len(targets)
                    targets.update(random.sample(available_nodes, min(extra_needed, len(available_nodes))))
            
            targets = list(targets)
            
            # validate
            if len(targets) < m:
                print(f"stop: link-selection only got {len(targets)} targets instead of {m}")
            
        elif model_name == 'optimization':
            # optimization model using actual cost function
            delta = model_params.get('delta', 10)
            new_pos = np.random.rand(2)
            
            # calculate euclidean distances
            euclidean_distances = {}
            for j in existing_nodes:
                dist = np.linalg.norm(new_pos - positions[j])
                euclidean_distances[j] = dist
            
            # find central node (cached to avoid repeated calculations)
            central_node = max(existing_nodes, key=lambda node: (G.degree(node), -node))
            
            # calculate network distances
            central_path_lengths = nx.single_source_shortest_path_length(G, central_node)
            network_distances = {}
            for j in existing_nodes:
                network_distances[j] = central_path_lengths.get(j, float('inf'))
            
            # calculate costs
            costs = {}
            for j in existing_nodes:
                costs[j] = delta * euclidean_distances[j] + network_distances[j]
            
            sorted_nodes = sorted(costs.items(), key=lambda x: x[1])
            targets = [node for node, cost in sorted_nodes[:min(m, len(existing_nodes))]]
            
            # validate  
            if len(targets) < m:
                print(f"stop: optimization model only got {len(targets)} targets instead of {m}")
            
            # store position for new node
            positions[new_node_id] = new_pos
        
        # record the degrees of target nodes before adding edges
        for target in targets:
            degree = G.degree(target)
            degree_counts[degree] += 1
            total_attachments += 1
        
        # record degree distribution before adding new node
        for node in existing_nodes:
            degree = G.degree(node)
            degree_node_counts[degree] += 1
        
        # add the new node and edges to the network
        G.add_node(new_node_id)
        for target in targets:
            G.add_edge(new_node_id, target)
            
            # update repeated_nodes for BA model
            if model_name == 'ba':
                # add the target node once more to repeated_nodes (its degree increased by 1)
                repeated_nodes.append(target)
        
        # for ba model, add new node m times to repeated_nodes (it has degree m)
        if model_name == 'ba':
            for _ in range(m):
                repeated_nodes.append(new_node_id)
        
        if progress and ((trial + 1) % 1000 == 0 or (trial + 1) == num_trials):
            pct = 100.0 * (trial + 1) / num_trials
            print(f"  trial {trial+1}/{num_trials} ({pct:.1f}%)")
    
    # calculate Π(k) = (attachments to degree k) / (number of nodes with degree k)
    # this gives the attachment probability per node of degree k
    pi_k = {}
    for k in degree_counts.keys():
        if degree_node_counts[k] > 0:
            pi_k[k] = degree_counts[k] / degree_node_counts[k]
        else:
            pi_k[k] = 0
    
    # also calculate normalized probability distribution for comparison
    pi_k_normalized = {k: count / total_attachments for k, count in degree_counts.items()}
    
    return pi_k, pi_k_normalized


# main execution, puting it all together
# parameters
n = 5000  # mmaller initial network
m = 1
num_trials = 5000  

# measure preferential attachment for each model
# with status updates
print("measuring ba model")
pi_ba, pi_ba_norm = measure_preferential_attachment(n, m, 'ba', {}, num_trials, seed=42, progress=True)

print("measuring preferential attachment for copying model")
pi_copying, pi_copying_norm = measure_preferential_attachment(n, m, 'copying', {'p': 0.5}, num_trials, seed=43, progress=True)

print("measuring preferential attachment for link-selection model")
pi_link, pi_link_norm = measure_preferential_attachment(n, m, 'link_selection', {}, num_trials, seed=44, progress=True)

print("measuring preferential attachment for optimization model (δ=10)")
pi_opt, pi_opt_norm = measure_preferential_attachment(n, m, 'optimization', {'delta': 10}, num_trials, seed=45, progress=True)

# create figure with four subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Preferential Attachment Functions Π(k) for Network Growth Models', fontsize=14, fontweight='bold', y=0.995)

# find global axis limits across all models
all_k_values = []
all_pi_values = []

# gather all k and pi values
for pi_data in [pi_ba, pi_copying, pi_link, pi_opt]:
    all_k_values.extend(pi_data.keys())
    all_pi_values.extend(pi_data.values())

# determine limits
max_k = max(all_k_values)
max_pi = max(all_pi_values)

# padding the limits
k_limit = max_k * 1.05
pi_limit = max_pi * 1.05

# plot 1: ba
ax = axes[0, 0]
k_values = sorted(pi_ba.keys())
pi_values = [pi_ba[k] for k in k_values]
ax.plot(k_values, pi_values, 'k-', marker='o', markersize=4, linewidth=2)
ax.set_xlabel('Degree k', fontsize=11)
ax.set_ylabel('Π(k)', fontsize=11)
ax.set_title('Barabási-Albert Model\n(N=5,000, m=1)', fontsize=12)
ax.set_xlim(0, k_limit)
ax.set_ylim(0, pi_limit)
ax.grid(True, alpha=0.3)

# plot 2: copying model
ax = axes[0, 1]
k_values = sorted(pi_copying.keys())
pi_values = [pi_copying[k] for k in k_values]
ax.plot(k_values, pi_values, 'k-', marker='o', markersize=4, linewidth=2)
ax.set_xlabel('Degree k', fontsize=11)
ax.set_ylabel('Π(k)', fontsize=11)
ax.set_title('Copying Model\n(N=5,000, m=1, p=0.5)', fontsize=12)
ax.set_xlim(0, k_limit)
ax.set_ylim(0, pi_limit)
ax.grid(True, alpha=0.3)

# plot 3: link-selection model
ax = axes[1, 0]
k_values = sorted(pi_link.keys())
pi_values = [pi_link[k] for k in k_values]
ax.plot(k_values, pi_values, 'k-', marker='o', markersize=4, linewidth=2)
ax.set_xlabel('Degree k', fontsize=11)
ax.set_ylabel('Π(k)', fontsize=11)
ax.set_title('Link-Selection Model\n(N=5,000, m=1)', fontsize=12)
ax.set_xlim(0, k_limit)
ax.set_ylim(0, pi_limit)
ax.grid(True, alpha=0.3)

# plot 4: optimization model
ax = axes[1, 1]
k_values = sorted(pi_opt.keys())
pi_values = [pi_opt[k] for k in k_values]
ax.plot(k_values, pi_values, 'k-', marker='o', markersize=4, linewidth=2)
ax.set_xlabel('Degree k', fontsize=11)
ax.set_ylabel('Π(k)', fontsize=11)
ax.set_title('Optimization Model\n(N=5,000, δ=10)', fontsize=12)
ax.set_xlim(0, k_limit)
ax.set_ylim(0, pi_limit)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assignment01_question03_part2_section_de_per_node.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure 1 saved as assignment01_question03_part2_section_de_per_node.png")

# second figure with normalized probabilities
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Normalized Attachment Probability Distribution (Π(k) = attachments/total)', fontsize=14, fontweight='bold', y=0.995)

# Plot 1: ba (normalized)
ax = axes2[0, 0]
k_values = sorted(pi_ba_norm.keys())
pi_values = [pi_ba_norm[k] for k in k_values]
ax.plot(k_values, pi_values, 'k-', marker='o', markersize=4, linewidth=2)
ax.set_xlabel('Degree k', fontsize=11)
ax.set_ylabel('Π(k)', fontsize=11)
ax.set_title('Barabási-Albert Model\n(N=5,000, m=1)', fontsize=12)
ax.grid(True, alpha=0.3)

# plot 2: copying model (normalized)
ax = axes2[0, 1]
k_values = sorted(pi_copying_norm.keys())
pi_values = [pi_copying_norm[k] for k in k_values]
ax.plot(k_values, pi_values, 'k-', marker='o', markersize=4, linewidth=2)
ax.set_xlabel('Degree k', fontsize=11)
ax.set_ylabel('Π(k)', fontsize=11)
ax.set_title('Copying Model\n(N=5,000, m=1, p=0.5)', fontsize=12)
ax.grid(True, alpha=0.3)

# plot 3: link-selection model (normalized)
ax = axes2[1, 0]
k_values = sorted(pi_link_norm.keys())
pi_values = [pi_link_norm[k] for k in k_values]
ax.plot(k_values, pi_values, 'k-', marker='o', markersize=4, linewidth=2)
ax.set_xlabel('Degree k', fontsize=11)
ax.set_ylabel('Π(k)', fontsize=11)
ax.set_title('Link-Selection Model\n(N=5,000, m=1)', fontsize=12)
ax.grid(True, alpha=0.3)

# plot 4: optimization model (normalized)
ax = axes2[1, 1]
k_values = sorted(pi_opt_norm.keys())
pi_values = [pi_opt_norm[k] for k in k_values]
ax.plot(k_values, pi_values, 'k-', marker='o', markersize=4, linewidth=2)
ax.set_xlabel('Degree k', fontsize=11)
ax.set_ylabel('Π(k)', fontsize=11)
ax.set_title('Optimization Model\n(N=5,000, δ=10)', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assignment01_question03_part2_section_de_normalized.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure 2 saved as assignment01_question03_part2_section_de_normalized.png")

print("\nDone.")