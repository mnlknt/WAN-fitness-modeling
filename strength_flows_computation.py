import pandas as pd
import networkx as nx
import pandas as pd
import numpy as np

def degrees_by_region(el, basins):
    
    G = nx.from_pandas_edgelist(el, source = 'basin1', target = 'basin2', create_using = nx.Graph)
    
    attribute_dict = dict(zip(basins['basin_id'], basins['regions']))
    nx.set_node_attributes(G, attribute_dict, name = 'region')

    regs = basins.regions.value_counts()
    
    k_regions = pd.DataFrame(columns = list(regs.index))
    
    for region in regs.index:
        temp = []
        
        for j in range(0, len(basins)):
            
            node_j = basins.loc[j]['basin_id']
            k_region = sum(1 for neighbor in G.neighbors(node_j) if G.nodes[neighbor]['region'] == region)
            reg_j = basins.loc[j]['regions']
            temp.append(k_region)

        k_regions['{}'.format(region)] = temp
        
    k_regions['node'] = basins['basin_id']
        
    return k_regions


def precompute_strengths_flows(inpath, outpath):

    open_links = pd.read_csv(inpath / 'opensky_el_final.csv')
    nodes = pd.concat([open_links.basin1, open_links.basin2], ignore_index=True)
    nodes = nodes.drop_duplicates().reset_index(drop=True)

    pax = pd.read_csv(inpath / 'opensky_nodes_final.csv')
    pax = pax[pax.basin_id.isin(nodes)]

    k_reg = degrees_by_region(open_links, pax)
    k_reg = pd.merge(k_reg, pax[['basin_id', 'regions']], left_on='node', right_on='basin_id', how='inner')

    regs = pax.regions.value_counts()

    dk = k_reg.melt(
        id_vars=['basin_id', 'regions'],
        value_vars=list(regs.index),
        var_name='continent',
        value_name='value'
    )

    # Fit coefficients for the "basins_all" power-law fit s ~ alpha * k^beta.
    # Estimated from a proprietary airport-pair traffic dataset (not included in
    # this repository). See Figure 1 B of Fischetti et al., arXiv:2601.13867,
    # for the fit.
    ALPHA_BASINS_ALL = 9.059244850109636
    BETA_BASINS_ALL  = 1.2239751990088177

    dk['s_reg'] = (dk['value'] ** BETA_BASINS_ALL) * ALPHA_BASINS_ALL

    dk = dk.rename(columns={'value': 'k_reg'})

    gs = dk.groupby(by='basin_id')['s_reg'].sum()
    gs = pd.DataFrame(gs).reset_index()

    gs = pd.merge(gs, pax[['basin_id', 'pax_strength']], on='basin_id', how='inner')
    gs['x'] = gs.pax_strength / gs.s_reg

    dk = pd.merge(dk, gs[['basin_id', 'x']], on='basin_id', how='left')
    dk['s_scaled'] = dk.x * dk.s_reg

    sopen = dk.copy()
    sopen = sopen.drop(columns=['k_reg', 'x', 's_reg'])

    sopen = sopen.pivot(
        index=['basin_id', 'regions'],
        columns='continent',
        values='s_scaled'
    ).reset_index()

    sopen = sopen.drop(columns=['regions'])
    sopen.to_csv(outpath / 'strength_by_region_opensky.csv', index=0)

    ds = pd.merge(sopen, pax[['basin_id', 'regions']], left_on='basin_id', right_on='basin_id', how='left')
    dg = ds.groupby('regions')[['Africa', 'Asia', 'Central America', 'East Asia', 'Europe',
                                 'Europe / Asia', 'Middle East', 'North America', 'Oceania',
                                 'South America', 'Southeast Asia']].sum()

    dg.to_csv(outpath / 'ws_regions.csv')

    return sopen, dg


def ls_computation(inpath, outpath):

    open_links = pd.read_csv(inpath / 'opensky_el_final.csv')
    pax = pd.read_csv(inpath / 'opensky_nodes_final.csv')

    # compute region flows for opensky data
    dof = pd.merge(open_links, pax[['basin_id', 'regions']], left_on=['basin1'], right_on=['basin_id'], how = 'left')
    dof = dof.drop(columns=['basin_id'])
    dof = dof.rename(columns={'regions':'region_source'})

    dof = pd.merge(dof, pax[['basin_id', 'regions']], left_on=['basin2'], right_on=['basin_id'], how = 'left')
    dof = dof.drop(columns=['basin_id'])
    dof = dof.rename(columns={'regions':'region_destination'})

    regs = pax.regions.value_counts()
    ls = pd.DataFrame(0, columns = regs.index, index = regs.index)
        
    for reg_i, reg_j in zip(dof.region_source, dof.region_destination):
        
        if(reg_i != reg_j):
        
            ls.loc[reg_i, reg_j] = ls.loc[reg_i, reg_j] + 1
            ls.loc[reg_j, reg_i] = ls.loc[reg_j, reg_i] + 1
            
        else:
            
            ls.loc[reg_i, reg_j] = ls.loc[reg_i, reg_j] + 1

    ls.to_csv(outpath / 'ls_regions.csv')

    return ls