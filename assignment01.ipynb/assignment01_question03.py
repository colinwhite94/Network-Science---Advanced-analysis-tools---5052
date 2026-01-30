import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.generators.classic import complete_graph

#readme: Network Generation and Growth, Barabási–Albert Preferential Attachment Model
# This code generates a Barab´asi-Albert network with N = 10^4 and m = 4.
# Starting with a fully connected 4-node network as initial condition. 
# 
# i. Plot the degree distribution at intermediate steps in the network’s growth.
#    Plot these degree distributions when the network has 10^2, 10^3, and 10^4 nodes.
# 
# iii. Measures and plots the average local clustering coefficient as a function of N .
# 
# iv. Following Figure 5.6a in Chapter 5, measure the degree dynamics of one of
# the initial nodes and of the nodes added to the network at times t = 100,
# t = 1, 000 and t = 5, 000

# barabási–Albert preferential attachment model implementation

# Note: initialization of `create_using` is handled inline where needed.

n = 10000 # nodes
m = 4 # edges to attach from a new node to existing nodes

# helper function to select random subset
def _random_subset(repeated_nodes, m, seed=None):
    """Select m unique random elements from repeated_nodes"""
    if seed is not None:
        random.seed(seed)
    targets = set()
    while len(targets) < m:
        x = random.choice(repeated_nodes)
        targets.add(x)
    return targets

def barabasi_albert_graph(n, m, seed=None, initial_graph=None, *, create_using=None):
    # ba preferential attachment implementation
    # a graph of n nodes is grown by attaching new nodes each with m
    # edges that are preferentially attached to existing nodes with high degree.

    # parameters
    # n: number of nodes
    # m: number of edges to attach from a new node to existing nodes
    # initial_graph: graph 
    # create_using: Graph constructor

    # returns
    # G: graph

    # confirm inputs
    if create_using is None:
        create_using = nx.Graph()
    else:
        create_using.clear()
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
        )
    
    # initialize the graph
    if initial_graph is None:
        # default initial graph : complete graph on m+1 nodes
        G = complete_graph(4, create_using)
    else:
        if len(initial_graph) < m or len(initial_graph) > n:
            raise nx.NetworkXError(
                f"Barabási–Albert initial graph needs between m={m} and n={n} nodes"
            )
        G = initial_graph.copy()

    # list of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # start adding the other n - m0 nodes.
    source = len(G)
    
    # randomly select one of the initial nodes to track
    initial_nodes = list(G.nodes())
    tracked_node = random.choice(initial_nodes)
    tracked_node_data = []  # store (num_nodes, degree) tuples
    
    # track nodes added at specific network sizes
    node_at_100 = None 
    node_at_1000 = None
    node_at_5000 = None
    tracked_node_100_data = []
    tracked_node_1000_data = []
    tracked_node_5000_data = []
    
    # initialize degree sequence storage
    degree_sequence01 = None
    degree_sequence02 = None
    degree_sequence03 = None
    average_clustering_values = [] # to store average clustering values at intervals of 100 nodes

    while source < n:
        # choose m unique nodes from the existing nodes
        # pick uniformly from repeated_nodes (the preferential part)
        targets = _random_subset(repeated_nodes, m, seed)
        # add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # and the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        
        # increment source node index
        source += 1
        
        # track the degree of the selected initial node
        tracked_degree = G.degree(tracked_node)
        tracked_node_data.append((source, tracked_degree))
        
        # select and start tracking node added at specific netowrk sizes
        if source == 100:
            node_at_100 = source - 1  # the node that was just added
        if source == 1000:
            node_at_1000 = source - 1
        if source == 5000:
            node_at_5000 = source - 1
            
        # track degrees of nodes added at specific sizes
        if node_at_100 is not None:
            tracked_node_100_data.append((source, G.degree(node_at_100)))
        if node_at_1000 is not None:
            tracked_node_1000_data.append((source, G.degree(node_at_1000)))
        if node_at_5000 is not None:
            tracked_node_5000_data.append((source, G.degree(node_at_5000)))
        
        # record average clustering coefficient every 100 nodes
        if source % 100 == 0:
            current_acc = nx.average_clustering(G, nodes=None, weight=None, count_zeros=True)
            average_clustering_values.append((source, current_acc))

        # capture snapshots at specific sizes
        if source == 100:
            degree_sequence01 = sorted((d for node, d in G.degree()), reverse=True)
        elif source == 1000:
            degree_sequence02 = sorted((d for node, d in G.degree()), reverse=True)
        elif source == 10000:
            degree_sequence03 = sorted((d for node, d in G.degree()), reverse=True)
            
    return (G, degree_sequence01, degree_sequence02, degree_sequence03, average_clustering_values, 
            tracked_node, tracked_node_data, node_at_100, tracked_node_100_data, 
            node_at_1000, tracked_node_1000_data, node_at_5000, tracked_node_5000_data)

# generate a single ba graph and capture snapshots
(G, degree_sequence01, degree_sequence02, degree_sequence03, average_clustering_values,
 tracked_node, tracked_node_data, node_at_100, tracked_node_100_data,
 node_at_1000, tracked_node_1000_data, node_at_5000, tracked_node_5000_data) = barabasi_albert_graph(n, m)

# create first figure with four subplots
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

# subplot 1, visualization (use the final graph at n=10000)
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw_networkx(G, pos, node_size=10, with_labels=False, node_color='dimgray', 
                 edge_color='lightgray', alpha=0.6, width=0.5, ax=ax1)
ax1.set_title(f'Barabási-Albert Graph\n(n=10000, m={m})')
ax1.axis('off')

#  subplot 2: degree distribution plot for n=100
ax2.plot(degree_sequence01, "k-", marker="o", markersize=4)
ax2.set_title("Degree Distribution Plot (n=100)")
ax2.set_ylabel("Degree")
ax2.set_xlabel("Distribution")
ax2.grid(True, alpha=0.3)

#  subplot 3: degree distribution plot for n=1000
ax3.plot(degree_sequence02, color='#505050', marker="o", markersize=3)
ax3.set_title("Degree Distribution Plot (n=1000)")
ax3.set_ylabel("Degree")
ax3.set_xlabel("Distribution")
ax3.grid(True, alpha=0.3)

#  subplot 4: degree distribution plot for n=10000
ax4.plot(degree_sequence03, color='#A0A0A0', marker="o", markersize=2)
ax4.set_title("Degree Distribution Plot (n=10000)")
ax4.set_ylabel("Degree")
ax4.set_xlabel("Distribution")
ax4.grid(True, alpha=0.3)

# adjust layout and save figure
fig1.tight_layout()
fig1.savefig('assignment01_question03_fig1.png')

# subplot 5: average clustering coefficient compared to number of nodes
fig3, (ax6, ax7) = plt.subplots(1, 2, figsize=(16, 6))
x_vals, y_vals = zip(*average_clustering_values)

#  subplot 6: linear scale
ax6.plot(x_vals, y_vals, "k-", marker="o", markersize=4)
ax6.set_title("Average Clustering Coefficient Over Time")
ax6.set_ylabel("Average Clustering Coefficient")
ax6.set_xlabel("Number of Nodes")
ax6.grid(True, alpha=0.3)

#  subplot 7: log scale on x-axis
ax7.plot(x_vals, y_vals, "k-", marker="o", markersize=4)
ax7.set_title("Average Clustering Coefficient Over Time (Log X-axis)")
ax7.set_ylabel("Average Clustering Coefficient")
ax7.set_xlabel("Number of Nodes (log scale)")
ax7.set_xscale('log')
ax7.grid(True, alpha=0.3)

# adjust layout and save figure
fig3.tight_layout()
fig3.savefig('assignment01_question03_fig3.png')

#  second   figure: combined degree distributions
fig2, (ax5, ax8) = plt.subplots(1, 2, figsize=(16, 6))

#  subplot 8: linear scale
ax5.plot(degree_sequence01, "k-", marker="o", markersize=3, label='n=100')
ax5.plot(degree_sequence02, color='#505050', marker="o", markersize=2, label='n=1000')
ax5.plot(degree_sequence03, color='#A0A0A0', marker="o", markersize=1, label='n=10000')
ax5.set_title("Combined Degree Distributions")
ax5.set_ylabel("Degree")
ax5.set_xlabel("Distribution")
ax5.legend()
ax5.grid(True, alpha=0.3)

#  subplot 9: log scale on x-axis
ax8.plot(degree_sequence01, "k-", marker="o", markersize=3, label='n=100')
ax8.plot(degree_sequence02, color='#505050', marker="o", markersize=2, label='n=1000')
ax8.plot(degree_sequence03, color='#A0A0A0', marker="o", markersize=1, label='n=10000')
ax8.set_title("Combined Degree Distributions (Log X-axis)")
ax8.set_ylabel("Degree")
ax8.set_xlabel("Distribution (log scale)")
ax8.set_xscale('log')
ax8.legend()
ax8.grid(True, alpha=0.3)

# adjust layout and save figure
fig2.tight_layout()
fig2.savefig('assignment01_question03_fig2.png')

#  third   figure: tracked node degree evolution
fig4, (ax9, ax10) = plt.subplots(1, 2, figsize=(16, 6))
num_nodes, degrees = zip(*tracked_node_data)

#  subplot 10: linear scale
ax9.plot(num_nodes, degrees, "k-", marker="o", markersize=2, label=f'Initial node {tracked_node}')
if tracked_node_100_data:
    num_nodes_100, degrees_100 = zip(*tracked_node_100_data)
    ax9.plot(num_nodes_100, degrees_100, color='#404040', marker="s", markersize=2, label=f'Node added at n=100')
if tracked_node_1000_data:
    num_nodes_1000, degrees_1000 = zip(*tracked_node_1000_data)
    ax9.plot(num_nodes_1000, degrees_1000, color='#707070', marker="^", markersize=2, label=f'Node added at n=1000')
if tracked_node_5000_data:
    num_nodes_5000, degrees_5000 = zip(*tracked_node_5000_data)
    ax9.plot(num_nodes_5000, degrees_5000, color='#A0A0A0', marker="d", markersize=2, label=f'Node added at n=5000')
ax9.set_title(f"Degree Evolution of Tracked Nodes")
ax9.set_ylabel("Number of Edges (Degree)")
ax9.set_xlabel("Number of Nodes in Graph")
ax9.legend()
ax9.grid(True, alpha=0.3)

#  subplot 11: log scale on both axes
ax10.plot(num_nodes, degrees, "k-", marker="o", markersize=2, label=f'Initial node {tracked_node}')
if tracked_node_100_data:
    num_nodes_100, degrees_100 = zip(*tracked_node_100_data)
    ax10.plot(num_nodes_100, degrees_100, color='#404040', marker="s", markersize=2, label=f'Node added at n=100')
if tracked_node_1000_data:
    num_nodes_1000, degrees_1000 = zip(*tracked_node_1000_data)
    ax10.plot(num_nodes_1000, degrees_1000, color='#707070', marker="^", markersize=2, label=f'Node added at n=1000')
if tracked_node_5000_data:
    num_nodes_5000, degrees_5000 = zip(*tracked_node_5000_data)
    ax10.plot(num_nodes_5000, degrees_5000, color='#A0A0A0', marker="d", markersize=2, label=f'Node added at n=5000')
ax10.set_title(f"Degree Evolution of Tracked Nodes (Log-Log Scale)")
ax10.set_ylabel("Number of Edges (Degree, log scale)")
ax10.set_xlabel("Number of Nodes in Graph (log scale)")
ax10.set_xscale('log')
ax10.set_yscale('log')
ax10.legend()
ax10.grid(True, alpha=0.3)

# adjust layout and save figure
fig4.tight_layout()
fig4.savefig('assignment01_question03_fig4.png')

# show all plots
plt.show()

#confirmation
print("done")