import networkx as nx
import numpy as np
import datasketch as ds
import random
from operator import itemgetter
from hash_table import HashTable
from scipy.sparse import identity


def get_events_from(filepath):
    """ get the events list from a file"""
    event_list = []
    nodelist = set()
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                pass
            line = line.strip().split()
            u, v, t = int(line[0]), int(line[1]), float(line[2])
            event_list.append((u, v, t))
            nodelist.add(u)
            nodelist.add(v)
    return event_list, list(nodelist)


def write_events(filepath, events):
    """write the list of events in filepath"""
    with open(filepath, 'w') as f:
        for event in events:
            string = str(event[0]) + ' ' + str(event[1]) + ' ' + str(event[2]) + '\n'
            f.write(string)
    return 0


def poisson_temporal_network(static_network, mean_iet, max_t, seed=None):
    """
    This function has been written by Arash Badie-Modiri
    """
    gen = np.random.default_rng(seed)
    events = []

    for i, j in static_network.edges():
        t = gen.exponential(mean_iet)  # residual
        while t <= max_t:
            events.append((i, j, t))
            t += gen.exponential(mean_iet)
    print("Number of events: ", len(events))
    events = sorted(events, key=itemgetter(2))
    return events


def out_component_size_node(event_list, nodelist):
    """ returns the out component size of each node of the temporal graph"""
    outsize_node = dict()
    out_comp_node = dict()
    for i in nodelist:
        out_comp_node[i] = set()
        out_comp_node[i].add(str(i).encode('utf8'))

    for event in reversed(event_list):
        i, j, t = event
        if i == j:
            continue

        a = out_comp_node[i]
        b = out_comp_node[j]
        c = a.union(b)

        out_comp_node[i] = c
        out_comp_node[j] = c
    for i in nodelist:
        outsize_node[i] = len(out_comp_node[i])

    return outsize_node


def out_component_size_node_ds(event_list, nodelist, p):
    """ returns an estimation of the out component size of each node of the temporal graph"""
    outsize_node = dict()
    out_comp_node = dict()
    for i in nodelist:                                   # for every node of the temporal graph
        out_comp_node[i] = ds.HyperLogLogPlusPlus(p=p)  # create a HLL++ structure, in constant space
        out_comp_node[i].update(str(i).encode('utf8'))   # initialize it with the node itself, doesn't increase space
    # complexity of the first loop is O(n) where n = number of nodes

    for event in reversed(event_list):  # look at every event in reverse time order
        i, j, t = event
        if i == j:
            continue

        a = out_comp_node[i]            # outcomponent of node i
        b = out_comp_node[j]            # outcomponent of node j
        a.merge(b)                      # "merge" is equivalent to "union" of outcomponents
        b.merge(a)                      # merging is done in constant time and space
    # complexity of the second loop is O(m) where m = number of events

    for i in nodelist:                              # for each node
        outsize_node[i] = out_comp_node[i].count()  # count how many elements are there in its outcomponent
    # complexity of the third loop is O(n) where n = number of nodes
    # overall time complexity is O(n+m). Usually, m >> n. So time complexity is O(m)
    # overall space complexity is O(n). We only have to store one HLL per node

    return outsize_node  # whole distribution of the outcomponents' size of the nodes of the temporal graph


def SI_process(events, starting_node, link_table):
    """ returns the out-component of an SI process on a list of events"""
    infected_nodes = set()
    infected_hashed_nodes = set()
    infected_nodes.add(link_table[starting_node])
    infected_hashed_nodes.add(starting_node)
    for e in events:
        u, v, t = e
        if u in infected_hashed_nodes:
            v_prime = [i for i, j in enumerate(link_table) if j == v]
            infected_hashed_nodes.add(v)
            for v in v_prime:
                infected_nodes.add(v)
        if v in infected_hashed_nodes:
            u_prime = [i for i, j in enumerate(link_table) if j == u]
            infected_hashed_nodes.add(u)
            for u in u_prime:
                infected_nodes.add(u)
    return infected_nodes


def SI_process_parallel(events, link_table, output_dimension):
    """ Computes the component matrix"""
    C = np.eye(output_dimension, dtype=bool)
    for e in events:
        u, v, t = e
        u_prime, v_prime = link_table[u], link_table[v]
        union = C[u_prime] | C[v_prime]
        C[u_prime] = union
        C[v_prime] = union
    return C


def SI_process_parallel_enumeration(events, link_table, output_dimension):
    """ Uses sets instead of matrix representation"""
    out_comp_node = dict()
    outsize_node = [0 for i in range(output_dimension)]
    for i in range(output_dimension):
        out_comp_node[i] = set()
        out_comp_node[i].add(i)
    for event in events:
        i, j, t = event
        if i == j:
            continue
        i, j = link_table[i], link_table[j]

        a = out_comp_node[i]
        b = out_comp_node[j]
        c = a.union(b)

        out_comp_node[i] = c
        out_comp_node[j] = c
    for i in range(output_dimension):
        for j in out_comp_node[i]:
            outsize_node[j] += 1

    return outsize_node


def SI_process_parallel_ds(events, link_table, output_dimension):
    """ Uses HLL instead of matrix representation"""
    C = [ds.HyperLogLog() for i in range(output_dimension)]
    for i in range(output_dimension):
        C[i].update((str(i)).encode('utf8'))
    for e in events:
        u, v, t = e
        u_prime, v_prime = link_table[u], link_table[v]
        C[u_prime].merge(C[v_prime])
        C[v_prime].merge(C[u_prime])
    return C


def SI_process_parallel_sparse(events, link_table, output_dimension):
    """ Uses sparse matrix representation"""
    C = identity(output_dimension, dtype=bool, format='lil')
    for e in events:
        u, v, t = e
        u_prime, v_prime = link_table[u], link_table[v]
        union = C[u_prime] + C[v_prime]
        C[u_prime] = union
        C[v_prime] = union
    return C


def hashing_events(events, hashtable, dimension):
    """ relabel the events' nodes"""
    new_events = []
    if isinstance(hashtable, list):
        for u, v, t in events:
            new_event = hashtable[u], hashtable[v], t
            new_events.append(new_event)
    else:  # if hashtable is of class HashTable
        for u, v, t in events:
            new_event = hashtable.hash32(u) % dimension, hashtable.hash32(v) % dimension, t
            new_events.append(new_event)
    return new_events


def hashing_graph(G, hashtable, dimension):
    """ return the graph with hashed nodes"""
    mapping = dict()
    if isinstance(hashtable, list):
        for i in range(G.number_of_nodes()):
            mapping[i] = hashtable[i]
    else:  # if hashtable is of class HashTable
        for i in range(G.number_of_nodes()):
            mapping[i] = hashtable.hash32(i) % dimension
    # hashed_graph = nx.relabel_nodes(G, mapping, copy=True)
    new_nodes = set([mapping[i] for i in range(G.number_of_nodes())])
    hashed_graph = nx.Graph()
    hashed_graph.add_nodes_from(new_nodes)
    for e in G.edges():
        i, j = e
        u, v = mapping[i], mapping[j]
        if u == v:
            continue
            # pass
        else:
            hashed_graph.add_edge(u, v)
    return hashed_graph


def link_table(hashtable, input_dimension, output_dimension):
    """ return the table between input and output (labels and hashed labels)"""
    tables = [-1 for j in range(input_dimension)]
    reverse_tables = [[] for i in range(output_dimension)]
    for j in range(input_dimension):
        tables[j] = hashtable.hash32(j) % output_dimension
        reverse_tables[tables[j]].append(j)
    return tables, reverse_tables


def intersection_list(sets1, sets2):
    """ return the list of the intersections of the sets of each list"""
    res = []
    for i in range(len(sets1)):
        inter = sets1[i].intersection(sets2[i])
        res.append(inter)
    return res


def aggreg(hashtables, hashtables_reverse, tensor_S_k, u):
    for k in range(len(hashtables)):
        link_table = hashtables[k]  # h
        S_k = tensor_S_k[k]  # S_k
        v = link_table[u]  # h(u)
        OC_v = [i for i, j in enumerate(S_k[v]) if j]
        if k == 0:
            OC_u = []
            for l in OC_v:
                OC_u += hashtables_reverse[k][l]
            OC_u = set(OC_u)
        else:
            current_oc = []
            for l in OC_v:
                current_oc += hashtables_reverse[k][l]
            current_oc = set(current_oc)
            OC_u = OC_u.intersection(current_oc)
    return len(OC_u)





