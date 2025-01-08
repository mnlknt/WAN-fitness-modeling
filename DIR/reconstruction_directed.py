import pandas as pd

from rec_functions import *

#Define input and output paths

input_path = '/mnt/TANK4TB/Network_reconstruction/Airport_reconstruction/Data_codes_final/DIR/Input/'
output_path = '/mnt/TANK4TB/Network_reconstruction/Airport_reconstruction/Data_codes_final/DIR/Output/'
     

# Implement bdgm or not? 'yes' or 'no'
bdgm = 'no'

# choose values for alpha and beta
alpha = 0.72 #1 or 0.72
beta = 0 #0 or 1

# how many networks in the ensemble?
N = 100

z0 = 0.00000000001

el = pd.read_csv(input_path + "201901_edgelist.csv", sep = ',')
airports = pd.read_csv(input_path + 'airport_info.csv', sep = ',')


if bdgm == 'yes':
    
    from bdgm_functions import *
    
    regions = pd.read_csv(input_path + 'country_regions.csv', sep = ';') 
    
    if (alpha == 1) & (beta == 0):
        
        model = 'bdgm'
        
    elif (alpha == 0.72) & (beta == 1):
        
        model = 'comb-bdgm'
    
    else:
        
        print('Wrong parameters chosen for bdgm!')
        

    try:
        sin_regions = pd.read_csv(output_path + 'sin_regions.csv', index_col = 0)
        sout_regions = pd.read_csv(output_path + 'sout_regions.csv', index_col = 0)
        el = pd.read_csv(output_path + 'edgelist-regions.csv', index_col = 0) 
    except FileNotFoundError:
        sin_regions, sout_regions, el =  strengths_by_region(el, airports, regions, output_path)      
    
    try:
        nodes = pd.read_csv(output_path + 'nodes.csv', index_col = 0)
    except FileNotFoundError:    
        nodes = node_prop(el, airports, output_path)
    
    try:
        reg_pairs = pd.read_csv(output_path + 'reg_pairs_Ls.csv', index_col = 0)
    except FileNotFoundError:
        reg_pairs = L_computation(nodes, regions, el, output_path)
    
    try:
        dz = pd.read_csv(output_path + 'z_{}.csv'.format(model), index_col = 0)
    except FileNotFoundError:    
        dz = z_computation_bdgm(nodes, reg_pairs, output_path, z0, sin_regions, sout_regions, alpha, beta, model)   
        
    try:
        pair_nodes = pd.read_csv(output_path + 'rec_variables_{}.csv'.format(model), index_col = 0)
    except FileNotFoundError:     
        pair_nodes = reconstruction_variables_bdgm(nodes, sin_regions, sout_regions, dz, alpha, beta, reg_pairs, output_path, model)
    
    ensembles_generation(N, pair_nodes, output_path)
    
elif bdgm == 'no':
 
    if (alpha == 1) & (beta == 0):
        
        model = 'dgm'
        
    elif (alpha == 1) & (beta == 1):
        
        model = 'dist-dgm'
        
    elif (alpha == 0.72) & (beta == 0):
        
        model = 'nlin-dgm'
        
    elif (alpha == 0.72) & (beta == 1):
        
        model = 'comb-dgm'
    
    else:
        
        print('Wrong parameters chosen for model!')
        
    try:
        nodes = pd.read_csv(output_path + 'nodes.csv', index_col = 0)
    except FileNotFoundError:    
        nodes = node_prop(el, airports, output_path)
    
    try:
        dz = pd.read_csv(output_path + 'z_{}.csv'.format(model), index_col = None, header = None)
        z = dz[0].item()
    except FileNotFoundError:    
        z = z_computation(nodes, output_path, z0, alpha, beta, model)  
        
    try:
        pair_nodes = pd.read_csv(output_path + 'rec_variables_{}.csv'.format(model), index_col = 0)
    except FileNotFoundError:     
        pair_nodes = reconstruction_variables(nodes, z, alpha, beta, output_path, model)
    
    ensembles_generation(N, pair_nodes, output_path, model)

### CHECK THE DISTANCE FUNCTION, IT SHOULD COMPUTE AND SAVE THE DISTANCES FILE, BUT THEN IT SHOULD BE ERASED FROM MEMORY, SINCE IT OCCUPIES TOO MUCH

# real 522243
# rec 522242.9999611009
