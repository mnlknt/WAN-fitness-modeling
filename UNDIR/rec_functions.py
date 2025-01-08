##### UNDIRECTED #####

import numpy as np
import pandas as pd
import networkx as nx
import random
import os, os.path
import itertools

def node_prop(G, savepath):
    
    s = [weg for (node, weg) in G.degree(weight='weight')]
    k = [val for (node, val) in G.degree()]
    n = [node for (node, val) in G.degree()]
    regs = [reg for (node, reg) in G.nodes(data = 'region')]
    
    nodes = pd.DataFrame()
    nodes['node'] = n
    nodes['k'] = k
    nodes['s'] = s
    nodes['region'] = regs
    
    nodes.to_csv(savepath + 'nodes.csv')
    
    return nodes


def haversine(basins, savepath):
    
    coords = basins[['basin_id', 'latitude', 'longitude']]
    coords.set_index('basin_id', inplace = True)
    coords = coords.apply(pd.to_numeric)
    
    # create dataframe to store distance values
    from itertools import product
    
    distances = pd.DataFrame(list(product(coords.index, coords.index)), columns = ['n1', 'n2'])
    
    a = np.deg2rad(coords.values)
    b = np.deg2rad(coords.values)
    
    # Get the differentiations
    diff_lat = np.subtract.outer(a[:,0], b[:,0]).flatten()
    diff_long = np.subtract.outer(a[:,1], b[:,1]).flatten()
    
    a_lat = np.multiply.outer(a[:,0], [1]*len(a)).flatten()
    a_long = np.multiply.outer(a[:,1], [1]*len(a)).flatten()
    
    b_lat = np.array(len(b)* list(b[:,0]))
    b_long = np.array(len(b)* list(b[:,1]))
    
    # Compute distance
    add0 = np.cos(a_lat) * np.cos(b_lat) * (np.sin(diff_long * 0.5) ** 2)
    d = (np.sin(diff_lat * 0.5) ** 2) +  add0
    
    # Get h and assign into dataframe
    h = 2 * 6371 * np.arcsin(np.sqrt(d))
    
    distances['distance'] = h
    distances.set_index(['n1', 'n2'], inplace = True)
        
        
    distances.to_csv(savepath + 'haversine_distances_basins.csv')
    
    return distances


# new distance function for computation without loops
def distance_func(distance):
    
    peak_dis = 50.024052999999995
    m = 0.00304593
    q = -0.013297
    x0 = -2.92694997e+03
    a = 7.06102656e-04
    b = -2.42702855e-03
    
    distance = pd.Series(distance)
    
    prob_dist = pd.Series([None] * len(distance))
    
    prob_dist[distance <= peak_dis] = m * distance[distance <= peak_dis] + q
    prob_dist[distance > peak_dis] = np.exp((x0 - distance[distance > peak_dis]) * a) - b
    
    return prob_dist

# TO FINISH
def equation_to_solve(z, nodes, alpha, beta, distances):
      
    prod = list(itertools.product(nodes.node, nodes.node))
    equal_indices = [i for i, (a, b) in enumerate(prod) if a == b]  #find indices of all elements where i = j
    
    str_prod = np.multiply.outer(nodes.s.to_numpy(), nodes.s.to_numpy()).flatten()
    str_prod = np.delete(str_prod, equal_indices)
    
    if beta == 1:
        
        temp_dist = distances.loc[prod, 'distance'].values
        temp_dist = temp_dist[temp_dist != 0]
        
        dist = distance_func(temp_dist)
            
    else:
        
        dist = [1]*len(str_prod)
        
    d = sum((np.power(str_prod, alpha) * np.power(dist, beta)) / (1 + z * np.power(str_prod, alpha) * np.power(dist, beta)))
    
    return d

# TO FINISH
def z_solution(a, L, nodes, alpha, beta, distances):
    eps = 1
    while(eps > 0.000000000000001):
        x = L/equation_to_solve(a, nodes, alpha, beta, distances)
        eps = x - a
        a = x
        
    return a

# TO FINISH
def z_computation(basins, nodes, savepath, z0, alpha, beta, model): # model is used to name saved file, can be:

    if beta == 1:
        
        distances = haversine(basins, savepath)
    else:
        
        distances = 1
        
    L = sum(nodes.k)
    
    z = z_solution(z0, L, nodes, alpha, beta, distances)
    
    with open(savepath + 'z_{}.csv'.format(model), 'w') as f_output:
        f_output.write(str(z))
    
    return z
       
            
def ensembles_generation(N, pair_nodes, savepath, model):
    
    columns = ['node_source', 'node_target', 'ws']
    
    for i in range(N):
        
        random_numbers = [random.random() for _ in range(len(pair_nodes))]
        pair_nodes['randoms'] = random_numbers
    
        mask = pair_nodes.randoms/pair_nodes.ps < 1 #where this is true I have a link
    
        df = pair_nodes[mask][columns]
        df.reset_index(drop = True, inplace = True)
        
        os.makedirs(os.path.dirname(savepath + 'Ensembles/{}/'.format(model)), exist_ok = True)
        df.to_csv(savepath + 'Ensembles/{}/edgelist_{}.csv'.format(model, i))


def reconstruction_variables(nodes, z, alpha, beta, savepath, model):
    
    pair_nodes = nodes[['node', 's']].merge(nodes[['node', 's']], how='cross', suffixes=('_source', '_target'))
    
    condition = pair_nodes['node_source'] == pair_nodes['node_target']
    pair_nodes = pair_nodes.loc[~condition]
    pair_nodes.reset_index(drop = True, inplace = True)
    
    # I have both links i-j and j-i, I need to remove one!
    pair_nodes = pair_nodes.loc[pd.DataFrame(np.sort(pair_nodes[['node_source','node_target']], axis=1), index = pair_nodes.index).drop_duplicates().index]
    pair_nodes.reset_index(drop = True, inplace = True)
    
    w = sum(nodes.s)
    
    if beta == 1:
        
        try: 
            distances = pd.read_csv(savepath + 'haversine_distances_basins.csv')
            distances.set_index(['n1', 'n2'], inplace = True)
        except FileNotFoundError:
            distances = haversine(nodes, savepath)
        
        combs = list(zip(pair_nodes['node_source'], pair_nodes['node_target']))
        temp_dist = distances.loc[combs, 'distance'].values
        dist = distance_func(temp_dist)
        
        pair_nodes['distance'] = dist
        
    else:
        
        pair_nodes['distance'] = 1
        
    
    pair_nodes['ps'] = (z * np.power(pair_nodes.s_source*pair_nodes.s_target, alpha) * np.power(pair_nodes.distance, beta))/(1 + z * np.power(pair_nodes.s_source*pair_nodes.s_target, alpha) * np.power(pair_nodes.distance, beta))
    pair_nodes = pair_nodes[pair_nodes.ps != 0]
    
    pair_nodes['ws'] = (pair_nodes.s_source*pair_nodes.s_target)/(w*pair_nodes.ps)
    
    pair_nodes.to_csv(savepath + 'rec_variables_{}.csv'.format(model))
    
    return pair_nodes

















