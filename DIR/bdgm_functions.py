##### DIRECTED #####

import numpy as np
import pandas as pd
import itertools

from rec_functions import *

# this function assigns a region to each airport and computes the in/out-strengths by region, saving them in a csv and returning them as dataframes

def strengths_by_region(el, airports, regions, savepath):
    
    airports['regions'] = [np.nan]*len(airports)

    regs = regions.region.value_counts()
    
    for region in regs.index:
        
        selected = regions[regions.region == region]
        m = airports.NAME_ENGLI.isin(selected.country)
        airports.loc[m, 'regions'] = region
        
        
    airports = airports.rename(columns = {'Airport Code': 'source'})
    el = el.merge(airports[['source', 'regions']], on = 'source', how = 'left')
    
    airports = airports.rename(columns = {'source': 'target'})
    el = el.rename(columns = {'regions': 'reg_s'})
    el = el.merge(airports[['target', 'regions']], on = 'target', how = 'left')
    el = el.rename(columns = {'regions': 'reg_t'})
        
    airports = airports.rename(columns = {'target':'Airport Code'})
                          
    # compute in-strength by region
    sin_reg = el.groupby(by = ['target', 'reg_s'])['weight'].sum()
    sin_reg = pd.DataFrame(sin_reg)
    sin_reg.reset_index(inplace = True)
    sin_reg.to_csv(savepath + 'sin_regions.csv')
    
    
    # compute the out-strength by region
    sout_reg = el.groupby(by = ['source', 'reg_t'])['weight'].sum()
    sout_reg = pd.DataFrame(sout_reg)
    sout_reg.reset_index(inplace = True)
    sout_reg.to_csv(savepath + 'sout_regions.csv')
    
    el.to_csv(savepath + 'edgelist-regions.csv')
    
    return sin_reg, sout_reg, el



# equations to use to compute z for bcgm


def L_computation(nodes, regions, el, savepath):
    
    import itertools
    
    regs = regions.region.value_counts()
    
    reg_pairs = list(itertools.product(regs.index, repeat=2))
    reg_pairs = pd.DataFrame(reg_pairs, columns = ['region_source', 'region_target'])
    
    Ls = []
    ws = []
    
    for reg_i, reg_j in zip(reg_pairs.region_source, reg_pairs.region_target):
        
        snodes_i = nodes[nodes.region == reg_i].reset_index(drop = True)
        snodes_j = nodes[nodes.region == reg_j].reset_index(drop = True)
        
        mask = el['source'].isin(snodes_i['node']) & el['target'].isin(snodes_j['node'])
        L = np.sum(mask) #get all links from region i to region j
        Ls.append(L)
        
        w = np.sum(el[mask].weight)
        ws.append(w)
        
    reg_pairs['Ls'] = Ls
    reg_pairs['Ws_tot'] = ws
    
    reg_pairs.to_csv(savepath + 'reg_pairs_Ls.csv')
    
    return reg_pairs
    
    
    

def equation_to_solve_bdgm(z, str_prod, dist, alpha, beta):
    
    d = sum((np.power(str_prod, alpha) * np.power(dist, beta)) / (1 + z * np.power(str_prod, alpha) * np.power(dist, beta)))
    
    return d

# CHECK THAT THE OUTER PRODUCT HAS SAME LENGTH OF THE DISTANCES


def z_solution_bdgm(a, L, str_prod, dist, alpha, beta):
    eps = 1
    while(eps > 0.000000000000001):
        x = L/equation_to_solve_bdgm(a, str_prod, dist, alpha, beta)
        eps = x - a
        a = x
        
    return a

# def z_solution_bdgm(a, L, str_prod, dist, alpha, beta):
#     for _ in range(500):
        
#         x = L/equation_to_solve_bdgm(a, str_prod, dist, alpha, beta)
#         a = x
        
#     return a



def z_computation_bdgm(nodes, reg_pairs, savepath, z0, sin_regions, sout_regions, alpha, beta, model):
    
    if beta == 1:
        
        try: 
            distances = pd.read_csv(savepath + 'haversine_distances_airports.csv')
            distances.set_index(['n1', 'n2'], inplace = True)
        except FileNotFoundError:
            distances = haversine(nodes, savepath)
    else:
        
        distances = 1
    
    
    region1 = []
    region2 = []
    zs = []
    
    for reg_i, reg_j, L in zip(reg_pairs.region_source, reg_pairs.region_target, reg_pairs.Ls):
        print('{}, {}'.format(reg_i, reg_j))
        
        snodes_i = nodes[nodes.region == reg_i].reset_index(drop = True)
        snodes_j = nodes[nodes.region == reg_j].reset_index(drop = True)
        
        if reg_i == reg_j:
           
           sin_region = sin_regions[(sin_regions.target.isin(snodes_j['node'])) & (sin_regions.reg_s == '{}'.format(reg_i))]
           sout_region = sout_regions[(sout_regions.source.isin(snodes_i['node'])) & (sout_regions.reg_t == '{}'.format(reg_j))]
           
           prod = list(itertools.product(sout_region.source, sin_region.target))
           equal_indices = [i for i, (a, b) in enumerate(prod) if a == b]  #find indices of all elements where i = j
           
           str_prod = np.multiply.outer(sout_region.weight.to_numpy(), sin_region.weight.to_numpy()).flatten()
           str_prod = np.delete(str_prod, equal_indices)
           
        else:
           
           sin_region = sin_regions[(sin_regions.target.isin(snodes_j['node'])) & (sin_regions.reg_s == '{}'.format(reg_i))]
           sout_region = sout_regions[(sout_regions.source.isin(snodes_i['node'])) & (sout_regions.reg_t == '{}'.format(reg_j))]
           
           str_prod = np.multiply.outer(sout_region.weight.to_numpy(), sin_region.weight.to_numpy()).flatten()
           
        if beta == 1:
           
           combs = list(itertools.product(sout_region.source, sin_region.target)) # all possible combinations of the nodes in i and j
           temp_dist = distances.loc[combs, 'distance'].values
           temp_dist = temp_dist[temp_dist != 0]
           
           dist = distance_func(temp_dist)
               
        else:
           
           dist = [1]*len(str_prod)        
        
        z = z_solution_bdgm(z0, L, str_prod, dist, alpha, beta)
        
        #print(z)
        region1.append(reg_i)
        region2.append(reg_j)
        zs.append(z)
            
    dz = pd.DataFrame()
    dz['region_source'] = region1
    dz['region_target'] = region2
    dz['z'] = zs
        
    dz.to_csv(savepath + 'z_{}.csv'.format(model))
    
    return dz
        

def reconstruction_variables_bdgm(nodes, sin_regions, sout_regions, dz, alpha, beta, reg_pairs, savepath, model):
    
    pair_nodes = nodes[['node', 'region']].merge(nodes[['node', 'region']], how='cross', suffixes=('_source', '_target'))
    
    condition = pair_nodes['node_source'] == pair_nodes['node_target']
    pair_nodes = pair_nodes.loc[~condition]
    pair_nodes.reset_index(drop = True, inplace = True)
    
    sin_regions = sin_regions.rename(columns = {'target':'node_target', 'reg_s':'region_source'})
    pair_nodes = pair_nodes.merge(sin_regions, on = ['node_target', 'region_source'], how  = 'left')
    pair_nodes = pair_nodes.rename(columns = {'weight': 'sin_region'})
    
    
    sout_regions = sout_regions.rename(columns = {'source':'node_source', 'reg_t':'region_target'})
    pair_nodes = pair_nodes.merge(sout_regions, on = ['node_source', 'region_target'], how  = 'left')
    pair_nodes = pair_nodes.rename(columns = {'weight': 'sout_region'})
    
    pair_nodes.dropna(inplace = True)
    pair_nodes.reset_index(drop = True, inplace = True)
    
    pair_nodes = pair_nodes.merge(dz, on = ['region_source', 'region_target'], how  = 'left')
    
    if beta == 1:
        
        try: 
            distances = pd.read_csv(savepath + 'haversine_distances_airports.csv')
            distances.set_index(['n1', 'n2'], inplace = True)
        except FileNotFoundError:
            distances = haversine(nodes, savepath)
        
        combs = list(zip(pair_nodes['node_source'], pair_nodes['node_target']))
        temp_dist = distances.loc[combs, 'distance'].values
        dist = distance_func(temp_dist)
        
        pair_nodes['distance'] = dist
        
    else:
        
        pair_nodes['distance'] = 1

    pair_nodes['ps'] = (pair_nodes.z * np.power(pair_nodes.sout_region*pair_nodes.sin_region, alpha) * np.power(pair_nodes.distance, beta))/(1 + pair_nodes.z * np.power(pair_nodes.sout_region*pair_nodes.sin_region, alpha) * np.power(pair_nodes.distance, beta))
    pair_nodes = pair_nodes.merge(reg_pairs[['region_source', 'region_target', 'Ws_tot']], on = ['region_source', 'region_target'], how  = 'left')
    pair_nodes['ws'] = (pair_nodes.sout_region*pair_nodes.sin_region)/(pair_nodes.Ws_tot*pair_nodes.ps)
    
    pair_nodes.to_csv(savepath + 'rec_variables_{}.csv'.format(model))
    
    return pair_nodes

    
        
        
        
        
