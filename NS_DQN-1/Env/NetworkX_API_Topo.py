import networkx as nx
from itertools import islice
import os


# Build a networkX graph with or without predefined link costs
def build_graph_from_txt(bw_capacity,weights=False):
    graph = nx.Graph()
    for (device1,device2) in bw_capacity:
        if not graph.has_node(device1):
            graph.add_node(device1)
        if not graph.has_node(device2):
            graph.add_node(device2)
        if weights:
            graph.add_edge(device1,device2,weight=bw_capacity[(device1,device2)])
        else:
            graph.add_edge(device1,device2,weight=1)
    return graph

# Get the k-shortest paths between each pair of hosts
def get_k_shortest_paths(graph, number_hosts, number_paths):
    paths = {}
    count=0
    number_hosts=number_hosts*number_hosts
    for src in graph:
        if "S" not in src:
            for dst in graph:
                if "S" not in dst:
                    paths[(src,dst)]=k_shortest_paths(graph,src,dst,number_paths)
                    for path in paths[(src, dst)]:
                        if len(path) != 0:
                            path.append(dst)
                            path.insert(0, src)
                    count=count+1
                    if number_hosts == count:
                        return paths
    return paths
            
def k_shortest_paths(graph, source, target, k):
    
    try: 
        calc = list(islice(nx.shortest_simple_paths(graph, source, target), k))
    except nx.NetworkXNoPath:
        calc = []
        
    return [path[1:-1] for path in calc]

# Run Dijkstra between two hosts
def dijkstra(graph, src, dst):
    
    path = nx.dijkstra_path(graph, src, dst)
    
    return path
    
# Add in and out ports to the path creating tuples ("switch", "in_port", "out_port")
def add_ports_to_path(path, adjacency, src_mac, dst_mac):

    p_tuples = []
    
    if len(path) < 4:
        p_tuples.append((path[1][1:], adjacency[(path[1][1:],src_mac)], 
                        adjacency[(path[1][1:],dst_mac)]))
        return p_tuples
    
    switches_in_path = path[1:-1]
    p_tuples.append((switches_in_path[0][1:], adjacency[(switches_in_path[0][1:],src_mac)], 
                    adjacency[(switches_in_path[0][1:], switches_in_path[1][1:])]))
    
    for i in range(1, len(switches_in_path)-1):
        p_tuples.append((switches_in_path[i][1:], adjacency[(switches_in_path[i][1:], switches_in_path[i-1][1:])], 
                        adjacency[(switches_in_path[i][1:], switches_in_path[i+1][1:])]))
        
    p_tuples.append((switches_in_path[-1][1:], adjacency[(switches_in_path[-1][1:], switches_in_path[-2][1:])], 
                    adjacency[(switches_in_path[-1][1:],dst_mac)]))
    
    return p_tuples

def convert_path_into_pairs(path):
    pairs=[]
    switches = [switch for switch in path]
    if len(switches) > 1:
        pairs = [(switches[i], switches[i + 1]) for i in range(len(switches) - 1)]
    return pairs


