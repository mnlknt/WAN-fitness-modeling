import pandas as pd

from rec_functions import *

#Define input and output paths

input_path = '/mnt/TANK4TB/Network_reconstruction/Airport_reconstruction/Data_codes_final/UNDIR/Input/'
output_path = '/mnt/TANK4TB/Network_reconstruction/Airport_reconstruction/Data_codes_final/UNDIR/Output/'
     

# Implement bdgm or not? 'yes' or 'no'
bdgm = 'no'

# choose values for alpha and beta
alpha = 0.80 #1 or 0.80
beta = 1 #0 or 1

# how many networks in the ensemble?
N = 1

z0 = 0.00000000001

el = pd.read_csv(input_path + "autumn_basin_flight_network.csv", sep = ',')
basins = pd.read_csv(input_path + 'basin_info_w_regions.csv', sep = ',')


if bdgm == 'yes':
    
    from bdgm_functions import *
    
    if (alpha == 1) & (beta == 0):
        
        model = 'bdgm'
        
    elif (alpha == 0.80) & (beta == 1):
        
        model = 'comb-bdgm'
    
    else:
        
        print('Wrong parameters chosen for bdgm!')
        

    try:
        s_regions = pd.read_csv(output_path + 's_regions.csv', index_col = 0)
        G = nx.read_gml(output_path + 'real_graph.gml')
    except FileNotFoundError:
        s_regions, G =  strengths_by_region(el, basins, output_path)      
    
    try:
        nodes = pd.read_csv(output_path + 'nodes.csv', index_col = 0)
    except FileNotFoundError:    
        nodes = node_prop(G, output_path)
    
    try:
        ls = pd.read_csv(output_path + 'ls_regions.csv', index_col = 0)
        ws = pd.read_csv(output_path + 'ws_regions.csv', index_col = 0)
    except FileNotFoundError:
        ls, ws = L_computation(G, basins, output_path)
    
    try:
        dz = pd.read_csv(output_path + 'z_{}.csv'.format(model), index_col = 0)
    except FileNotFoundError:    
        dz = z_computation_bdgm(basins, output_path, z0, s_regions, alpha, beta, model, ls)   
        
    try:
        pair_nodes = pd.read_csv(output_path + 'rec_variables_{}.csv'.format(model), index_col = 0)
    except FileNotFoundError:     
        pair_nodes = reconstruction_variables_bdgm(nodes, s_regions, dz, alpha, beta, ws, output_path, model)
    
    ensembles_generation(N, pair_nodes, output_path, model)
    
elif bdgm == 'no':
 
    if (alpha == 1) & (beta == 0):
        
        model = 'dgm'
        
    elif (alpha == 1) & (beta == 1):
        
        model = 'dist-dgm'
        
    elif (alpha == 0.80) & (beta == 0):
        
        model = 'nlin-dgm'
        
    elif (alpha == 0.80) & (beta == 1):
        
        model = 'comb-dgm'
    
    else:
        
        print('Wrong parameters chosen for model!')
    
    try:
        nodes = pd.read_csv(output_path + 'nodes.csv', index_col = 0)
    except FileNotFoundError:    
        nodes = node_prop(G, output_path)
    
    try:
        dz = pd.read_csv(output_path + 'z_{}.csv'.format(model), index_col = None, header = None)
        z = dz[0].item()
    except FileNotFoundError:    
        z = z_computation(basins, nodes, output_path, z0, alpha, beta, model)   
        
    try:
        pair_nodes = pd.read_csv(output_path + 'rec_variables_{}.csv'.format(model), index_col = 0)
    except FileNotFoundError:     
        pair_nodes = reconstruction_variables(nodes, z, alpha, beta, output_path, model)
    
    ensembles_generation(N, pair_nodes, output_path, model)
    
### CHECK THE DISTANCE FUNCTION, IT SHOULD COMPUTE AND SAVE THE DISTANCES FILE, BUT THEN IT SHOULD BE ERASED FROM MEMORY, SINCE IT OCCUPIES TOO MUCH
