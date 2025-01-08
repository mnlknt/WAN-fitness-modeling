##### UNDIRECTED #####

import numpy as np
import pandas as pd
import itertools
import networkx as nx

from rec_functions import *

# this function assigns a region to each airport and computes the strengths by region, saving them in a csv and returning them as dataframes

def strengths_by_region(el, basins, savepath):
    
    G = nx.from_pandas_edgelist(el, source = 'basin_id1', target = 'basin_id2', edge_attr = ["weight"], create_using = nx.Graph)
    
    attribute_dict = dict(zip(basins['basin_id'], basins['regions']))
    nx.set_node_attributes(G, attribute_dict, name = 'region')

    regs = basins.regions.value_counts()
    
    s_regions = pd.DataFrame(columns = list(regs.index))
    
    for region in regs.index:
        temp = []
        
        for j in range(0, len(basins)):
            
            node_j = basins.loc[j]['basin_id']
            s_region = sum(G.edges[neighbor, node_j]['weight'] for neighbor in G.neighbors(node_j) if G.nodes[neighbor]['region'] == region)
            reg_j = basins.loc[j]['regions']
           
            temp.append(s_region)
        s_regions['{}'.format(region)] = temp
        
    s_regions['node'] = basins['basin_id']
    
    s_regions.to_csv(savepath + 'strengths_regions.csv') 
    nx.write_gml(G, savepath + 'real_graph.gml')
    
    return s_regions, G


# equations to use to compute z for bcgm

def L_computation(G, basins, savepath):
        
     regs = basins.regions.value_counts()
    
     ls = pd.DataFrame(0, columns = regs.index, index = regs.index)
     ws = pd.DataFrame(0, columns = regs.index, index = regs.index)
    
     for edge in G.edges:
        
        node_i = edge[0]
        node_j = edge[1]
        
        reg_i = G.nodes[node_i]['region']
        reg_j = G.nodes[node_j]['region']
        
        weight = G.edges[node_i, node_j]['weight']
        
        if(reg_i != reg_j):
        
            ls.loc[reg_i, reg_j] = ls.loc[reg_i, reg_j] + 1
            ls.loc[reg_j, reg_i] = ls.loc[reg_j, reg_i] + 1
            
            ws.loc[reg_i, reg_j] = ws.loc[reg_i, reg_j] + weight
            ws.loc[reg_j, reg_i] = ws.loc[reg_j, reg_i] + weight
        
        else:
            
            ls.loc[reg_i, reg_j] = ls.loc[reg_i, reg_j] + 1
            ws.loc[reg_i, reg_j] = ws.loc[reg_i, reg_j] + weight
            
     ls.to_csv(savepath + 'ls_regions.csv')
     ws.to_csv(savepath + 'ws_regions.csv')
    
     return ls, ws


def equation_to_solve_bdgm(z, multiplication, alpha, beta, dist, reg_i, reg_j):
    
    if reg_i == reg_j:
        d = sum((np.power(multiplication, alpha) * np.power(dist, beta)) / (1 + z * np.power(multiplication, alpha) * np.power(dist, beta)))/2
    else:   
        d = sum((np.power(multiplication, alpha) * np.power(dist, beta)) / (1 + z * np.power(multiplication, alpha) * np.power(dist, beta)))
    
    return d


def z_solution_bdgm(a, L, multiplication, alpha, beta, dist, reg_i, reg_j):
    eps = 1
    while(eps > 0.000000000000001):
        x = L/equation_to_solve_bdgm(a, multiplication, alpha, beta, dist, reg_i, reg_j)
        eps = x - a
        a = x
        
    return a


def z_computation_bdgm(basins, savepath, z0, s_regions, alpha, beta, model, ls):

    regs = basins.regions.value_counts()
    reg_pairs = list(itertools.product(regs.index, regs.index))
    
    dr = pd.DataFrame(reg_pairs, columns=['region_1', 'region_2'])
    dr = dr.loc[pd.DataFrame(np.sort(dr[['region_1','region_2']], axis = 1), index = dr.index).drop_duplicates().index]
    dr.reset_index(drop = True, inplace = True)
    
    zs = []
    
    for reg_i, reg_j in zip(dr.region_1, dr.region_2):

        print(reg_i, reg_j)
        
        L = ls.loc[reg_i, reg_j]
        
        snodes_i = basins[basins.regions == reg_i]['basin_id'].reset_index(drop = True)
        snodes_j = basins[basins.regions == reg_j]['basin_id'].reset_index(drop = True)
        
        sin_sector = s_regions['{}'.format(reg_i)][s_regions.node.isin(snodes_j)]
        sout_sector = s_regions['{}'.format(reg_j)][s_regions.node.isin(snodes_i)]
        
        if(reg_i != reg_j):
    
            multiplication = np.multiply.outer(sin_sector.to_numpy(), sout_sector.to_numpy()).flatten()
           
        else:
            
            multiplication = np.multiply.outer(sin_sector.to_numpy(), sout_sector.to_numpy())
            np.fill_diagonal(multiplication, 0)
            multiplication = multiplication.flatten()            
            
        if beta == 1:

            try: 
                distances = pd.read_csv(savepath + 'haversine_distances_basins.csv')
                distances.set_index(['n1', 'n2'], inplace = True)
            except FileNotFoundError:
                distances = haversine(basins, savepath)

           
            combs = list(itertools.product(snodes_j, snodes_i)) # all possible combinations of the nodes in i and j
            temp_dist = distances.loc[combs, 'distance'].values
           
            dist = distance_func(temp_dist)
               
        else:
           
            dist = [1]*len(multiplication)
        
        z = z_solution_bdgm(z0, L, multiplication, alpha, beta, dist, reg_i, reg_j)
        zs.append(z)
    
    dr['zs'] = zs

    dr.to_csv(savepath + 'z_{}.csv'.format(model))    
    
    return dr
        

def reconstruction_variables_bdgm(nodes, s_regions, dz, alpha, beta, ws, savepath, model):
    
    pair_nodes = nodes[['node', 'region']].merge(nodes[['node', 'region']], how='cross', suffixes=('_source', '_target'))
    
    condition = pair_nodes['node_source'] == pair_nodes['node_target']
    pair_nodes = pair_nodes.loc[~condition]
    pair_nodes.reset_index(drop = True, inplace = True)
    
    # I have both links i-j and j-i, I need to remove one!
    pair_nodes = pair_nodes.loc[pd.DataFrame(np.sort(pair_nodes[['node_source','node_target']], axis=1), index = pair_nodes.index).drop_duplicates().index]
    pair_nodes.reset_index(drop = True, inplace = True)
    
    s_regions.set_index('node', inplace = True)

    s_reg1 = []
    for reg, t in zip(pair_nodes.region_source, pair_nodes.node_target):
        s_reg1.append(s_regions.loc[t,reg])
        
    s_reg2 = []
    for reg, t in zip(pair_nodes.region_target, pair_nodes.node_source):
        s_reg2.append(s_regions.loc[t,reg])
    
    pair_nodes['s_reg_source'] = s_reg1
    pair_nodes['s_reg_target'] = s_reg2
    
    dz2 = dz.copy()
    dz2.rename(columns = {'region_1':'region_2', 'region_2':'region_1'}, inplace = True)
    
    zstot = pd.concat([dz, dz2], ignore_index = True)
    zstot.drop_duplicates(inplace = True)
    zstot.rename(columns = {'region_1':'region_source', 'region_2':'region_target'}, inplace = True)
    
    pair_nodes = pair_nodes.merge(zstot, on = ['region_source', 'region_target'], how  = 'left')
    
    wtot = ws.stack()
    wtot = pd.DataFrame(wtot)
    wtot.reset_index(inplace = True)
    wtot.rename(columns = {'level_0':'region_source', 'level_1':'region_target', 0:'Ws_tot'}, inplace = True)
        
    pair_nodes = pair_nodes.merge(wtot, on = ['region_source', 'region_target'], how  = 'left')
    
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
        
    
    pair_nodes['ps'] = (pair_nodes.zs * np.power(pair_nodes.s_reg_source*pair_nodes.s_reg_target, alpha) * np.power(pair_nodes.distance, beta))/(1 + pair_nodes.zs * np.power(pair_nodes.s_reg_source*pair_nodes.s_reg_target, alpha) * np.power(pair_nodes.distance, beta))
    pair_nodes = pair_nodes[pair_nodes.ps != 0]
    
    pair_nodes['ws'] = (pair_nodes.s_reg_source*pair_nodes.s_reg_target)/(pair_nodes.Ws_tot*pair_nodes.ps)
    
    mask = pair_nodes.region_source == pair_nodes.region_target
    pair_nodes.loc[mask,'ws'] = pair_nodes.loc[mask,'ws']/2
    
    pair_nodes.to_csv(savepath + 'rec_variables_{}.csv'.format(model))
    
    return pair_nodes
