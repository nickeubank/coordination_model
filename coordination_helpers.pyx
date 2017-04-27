
import numpy as np
cimport numpy as np
import numpy.random as npr

cpdef cython_iterate(np.ndarray participating, np.ndarray local_avg, np.ndarray beta, object graph):
    cdef int graph_size

    graph_size = graph.vcount() 

    for v in npr.permutation(range(graph_size)):
        neighbor_indices = get_neighbors(v, graph)

        if len(neighbor_indices) > 0:
            local_avg[v] = np.mean( participating[neighbor_indices] )
            participating[v] = (beta[v] - (1 - local_avg[v]) > 0)
    
    return participating, local_avg

cdef list get_neighbors(int v, object graph):
    cdef int index
    cdef object i
    cdef list neighbor_indices

    neighbor_indices = list()
    for i in graph.vs[v].neighbors():
        index = i.index
        neighbor_indices.append(index)
       
    return neighbor_indices
