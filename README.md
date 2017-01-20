This project contains code for running a coordination game simulation based on David Siegel (2009)'s "Social Networks and Collective Action" in the American Journal of Political Science. 

All relevant code is contained in `coordination_model.py`, and simulations are 
started using the `run_coordination_simulation` function. 

Returns pandas Series where each entry is share of people coordinated on candidate 1 at end of 
each simulation run. 

The one potential divergence from Siegel's model is that I do not update individual nodes simultaneously. Siegel's writing suggests that all nodes examine behavior in their neighborhood, then update simultaneously. This, I discovered, leads to what I feel is a pathological dynamic that is knife-edge dependent on this simultaneity. The simplest illustration of this is when a network has only two nodes, $i$, and $j$, where $i$ is in state 0 and $j$ is in state 1. If neither have strong dispositions towards one state, then what happens is each looks at their neighborhood, finds everyone is in the alternate state, then they \emph{simultaneously} flip states. This flipping back and forth can go on forever, and I've found it often happens in much larger groups than just two nodes. Since any deviation from simultaneity breaks this pattern, it seems unrealistic, and so instead I moved the model to randomly iterate through nodes, updating each node sequentially. This change was made with commit 6d2ef1bc19eb3, and interested parties can revert to an earlier commit if they wish to work with simultaneous updating. 