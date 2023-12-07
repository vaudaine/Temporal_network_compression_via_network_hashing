import utils
import networkx as nx
from hash_table import HashTable
import multiprocessing as mp
import ot
import time
import numpy as np

stop_warning = True
if stop_warning:
    import warnings
    warnings.filterwarnings("ignore")

if __name__ == '__main__':
    number_of_hashtables = 5
    number_of_nodes = 10000
    number_supernodes = 5000
    G = nx.gnp_random_graph(n=number_of_nodes, p=1.5/1000, seed=42)  # underlying static graph
    events = utils.poisson_temporal_network(G, mean_iet=1, max_t=5)  # list of events (u, v, t)

    #############
    # No hashing part
    #############

    # Parameters without hashing
    link_table = [i for i in range(number_of_nodes)]  # identity
    output_dimension = number_of_nodes  # we don't change anything

    # Computation of the component matrix
    start = time.time()
    S = utils.SI_process_parallel(events, link_table, output_dimension)
    print("Time for component matrix computation: " + str(time.time() - start))
    # you may want to include the computation of the sizes in the timer too

    # Computation with the HLL algorithm
    S_hll = utils.SI_process_parallel_ds(events, link_table, output_dimension)

    # Computation with a sparse representation
    S_sparse = utils.SI_process_parallel_sparse(events, link_table, output_dimension)

    # Computation of the size of the out-components
    OC_size = np.sum(S, axis=0)
    OC_size_ds = [S_hll[i].count() for i in range(number_of_nodes)]
    OC_size_sparse = S_sparse.sum(axis=0)

    # Histogram of the out-components sizes
    counts_0 = [0 for i in range(number_of_nodes + 1)]
    for i, j in enumerate(OC_size):
        counts_0[j] += 1 / number_of_nodes  # len(OC_size) = number_of_nodes

    ##########
    # Hashing part
    ##########

    print('Computing hash tables')
    tables = [HashTable(order=100, seed=420+i, multiseed=False) for i in range(number_of_hashtables)]
    hashtables, hashtables_reverse = [], []
    for i in range(number_of_hashtables):
        h, rh = utils.link_table(tables[i], number_of_nodes, number_supernodes)
        hashtables.append(h)
        hashtables_reverse.append(rh)
    # can be done in parallel too
    print("Hash tables done")

    args = ((events, hashtables[k], number_supernodes) for k in range(number_of_hashtables))
    with mp.Pool(mp.cpu_count()) as pool:
        tensor_S_k = pool.starmap(utils.SI_process_parallel, args)

    args = ((hashtables, hashtables_reverse, tensor_S_k, u) for u in range(number_of_nodes))
    with mp.Pool(mp.cpu_count()) as pool:
        OC_size_hash = pool.starmap(utils.aggreg, args)

    # Histogram of the approximated out-components sizes
    counts = [0 for i in range(number_of_nodes + 1)]
    for i, j in enumerate(OC_size_hash):
        counts[j] += 1 / number_of_nodes

    # Earth Movers distance between non-hashed and hashed histograms of out-components sizes
    dist = ot.emd2_1d(counts_0, counts)
    print("Distance between the true distribution and the hashed distribution of out-components sizes: " + str(dist))




