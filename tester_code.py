############################
# Code to test coordination game.
############################

import os
os.chdir('/users/nick/dropbox/GAPP/02_Main Evaluation/Activities/18_voting_and_networks/2_code/coordination_model_folder')
import coordination_model as cm
import igraph as ig


# Toy
g = ig.Graph()
g.add_vertices(4)
g.add_edges([(0,1), (1,2), (2,0)])

#a = cm.run_coordination_simulation(g, num_runs = 5, debug=True)
#print(a)

# Bigger random 
g2 = ig.Graph.Erdos_Renyi(n=20, p=0.1)
a = cm.run_coordination_simulation(g2, num_runs=5, debug=True)
print(a)
#cm.test_suite()
