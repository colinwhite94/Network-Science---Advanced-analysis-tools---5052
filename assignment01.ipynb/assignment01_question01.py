import networkx as nx
from networkx import watts_strogatz_graph
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# readme: Watts & Strogatz, 1998 figure 2 recreation
# this code generates Watts-Strogatz graphs with varying rewiring probabilities
# then computes the normalized average shortest path lengths and clustering coefficients
# then plots path lenths and clustering coefficients with their respective rewiring probability on a logarithmic scale

# each graph is generated 20 times for each p value and averages the results
# using 1000 nodes and 10 edges per node

# set parameters
n = 1000 # nodes
k = 10 # edges per node
p = 0.0001 # probability parameter of rewiring each edge
asp_tally = 0 # average shortest path length tally
aspR_tally = 0 # average shortest path length tally for regular graph
cc_tally = 0 # clustering coefficient tally
ccR_tally = 0 # clustering coefficient tally for regular graph
results01 = [] # to store normalized average shortest path lengths
results02 = [] # to store normalized clustering coefficients

# make callable regular/modified graphs
# modified graph
def ws(n, k, p):
    G = watts_strogatz_graph(n, k, p, seed=None, create_using=None)
    return G   
 
# regular graph
def wsR(n, k):
    GR = watts_strogatz_graph(n, k, 0, seed=None, create_using=None)
    return GR

# calculate average shortest path lengths
# average shortest path length of modified graph
def asp(G):
    asp_value = nx.average_shortest_path_length(G, weight=None, method=None)
    return asp_value

# average shortest path length of regular graph
def aspR(GR):
    aspR_value = nx.average_shortest_path_length(GR, weight=None, method=None)
    return aspR_value

# vary p from 0.0001 to 1
for j in range(15):
    
    # run 20 times and average the normalized average shortest path lengths
    for i in range(20):
        G = ws(n, k, p)
        GR = wsR(n, k)

        # average shortest path length calculations
        asp_current_value = asp(G)
        aspR_current_value = aspR(GR)
        asp_tally += asp_current_value
        aspR_tally += aspR_current_value

        # clustering coefficient calculations
        cc_current = nx.average_clustering(G, nodes=None, weight=None, count_zeros=True)
        ccR_current = nx.average_clustering(GR, nodes=None, weight=None, count_zeros=True)
        cc_tally += cc_current
        ccR_tally += ccR_current

        # increment
        i += 1
   
    # normalize the values
    normalized_asp = (asp_tally/aspR_tally)
    normalized_cc = (cc_tally/ccR_tally)
    # store results
    results01.append((p, normalized_asp))
    results02.append((p, normalized_cc))
    
    # increment p value and reset tallies
    p *= 1.8
    asp_tally = 0
    aspR_tally = 0
    cc_tally = 0
    ccR_tally = 0

    # increment
    j += 1

# plot results on same graph
# extract data for plotting
ps = [result[0] for result in results01]
normalized_asps = [result[1] for result in results01]

# extract clustering coefficients
cc_results = [result[1] for result in results02]
normalized_ccs = [result[1] for result in results02]

# create plot
plt.plot(ps, normalized_asps, marker='o', color='black', label='L(p)')
plt.plot(ps, normalized_ccs, marker='o', color='grey', label='C(p)')
plt.legend(['L(p)/L(0)', 'C(p)/C(0)'])
plt.xscale('log')

# set to 4 decimal places on x-axis
ax = plt.gca()
from matplotlib.ticker import FuncFormatter, FixedLocator
import numpy as np

# set x-axis limits
ax.set_xlim(right=1)

# set tick locations including 1
tick_locations = list(np.logspace(-4, 0, 5))  # creates ticks from 0.0001 to 1
ax.xaxis.set_major_locator(FixedLocator(tick_locations))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.4f}'))

# labels and title
plt.ylabel('Normalized Values')
plt.xlabel('Rewiring Probability (p)')
plt.title('Characteristic path length L(p) and clustering coefficient C(p)\nfor the family of randomly rewired graphs', fontsize=10)
plt.grid(True)
#save image
plt.tight_layout()
plt.savefig('assignment01_question01.png')  
plt.show()        



