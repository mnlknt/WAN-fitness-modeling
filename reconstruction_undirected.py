import pandas as pd
from pathlib import Path

from rec_functions import *
from strength_flows_computation import *

#Define input and output paths

BASE_DIR = Path.cwd()

INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)  # create it if it doesn't exist yet

inpath = Path(INPUT_DIR)
outpath = Path(OUTPUT_DIR)


# Which model to implement? 

model = 'model_B' # one of: 'model_K', 'model_S', 'model_D', 'model_B', 'model_C'

density = 'basins' #'basins' to use real density of basins filtered dataset, 'opensky' to use opensky density

# how many networks in the ensemble?
N = 10

z0 = 0.00000000001

try:
    el = pd.read_csv(inpath / "opensky_el_final.csv", sep = ',')
    basins = pd.read_csv(inpath / 'opensky_nodes_final.csv', sep = ',')
except FileNotFoundError as e:
    print(f"File not found: {e.filename}")
    print("Make sure the required input CSVs are in:", INPUT_DIR)

basins.basin_id = basins.basin_id.astype(int)
    
if density == 'opensky':

    if (model == 'model_B' or model == 'model_C'):
        
        from bdgm_functions import *
        
        if (model == 'model_B'):
            
            alpha = 1
            beta = 0
            
        elif (model == 'model_C'):
            
            alpha = 1.56
            beta = 1
        
        else:
            
            print('Wrong name chosen for model!')
            
        try:
            s_regions, ws_regions = precompute_strengths_flows(inpath, outpath)
        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")
            print("Make sure the required input CSVs are in:", INPUT_DIR)
        
        try:
            ls = pd.read_csv(outpath / 'ls_regions.csv', index_col = 0)
        except FileNotFoundError:    
            ls = ls_computation(inpath, outpath)   

        try:
            dz = pd.read_csv(outpath / 'z_{}.csv'.format(model), index_col = 0)
        except FileNotFoundError:    
            dz = z_computation_bdgm(basins, outpath, z0, s_regions, alpha, beta, model, ls)   
            
        try:
            pair_nodes = pd.read_csv(outpath / 'rec_variables_{}.csv'.format(model), index_col = 0)
        except FileNotFoundError:     
            pair_nodes = reconstruction_variables_bdgm(basins, s_regions, dz, alpha, beta, ws_regions, outpath, model)
        
        ensembles_generation(N, pair_nodes, outpath, model)
        
    elif (model == 'model_K' or model == 'model_S' or model == 'model_D'):
    
        if (model == 'model_K'):
            alpha = 1
            beta = 0
            
        elif (model == 'model_D'):
            
            alpha = 1
            beta = 1
            
        elif (model == 'model_S'):
            
            alpha = 0.80
            beta = 0
        
        else:
            
            print('Wrong parameters chosen for model!')
        
        L = len(el)

        try:
            dz = pd.read_csv(outpath / 'z_{}.csv'.format(model), index_col = None, header = None)
            z = dz[0].item()
        except FileNotFoundError:    
            z = z_computation(basins, outpath, z0, alpha, beta, model, L)   
            
        try:
            pair_nodes = pd.read_csv(outpath / 'rec_variables_{}.csv'.format(model), index_col = 0)
        except FileNotFoundError:     
            pair_nodes = reconstruction_variables(basins, z, alpha, beta, outpath, model)
        
        ensembles_generation(N, pair_nodes, outpath, model)

elif(density == 'basins'):
    
    if (model == 'model_B' or model == 'model_C'):
        
        from bdgm_functions import *
        
        if (model == 'model_B'):
            
            alpha = 1
            beta = 0
            
        elif (model == 'model_C'):
            
            alpha = 1.56
            beta = 1
        
        else:
            
            print('Wrong name chosen for model!')
            

        try:
            s_regions, ws_regions = precompute_strengths_flows(inpath, outpath)
        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")
            print("Make sure the required input CSVs are in:", INPUT_DIR)
        
        try:
            ls = pd.read_csv(outpath / 'ls_regions.csv', index_col = 0)
            ls = ls*(0.109/0.067)
        except FileNotFoundError:    
            ls = ls_computation(inpath, outpath)  
            ls = ls*(0.109/0.067) 
  
        try:
            dz = pd.read_csv(outpath / 'z_{}.csv'.format(model), index_col = 0)
        except FileNotFoundError:    
            dz = z_computation_bdgm(basins, outpath, z0, s_regions, alpha, beta, model, ls)   
            
        try:
            pair_nodes = pd.read_csv(outpath / 'rec_variables_{}.csv'.format(model), index_col = 0)
        except FileNotFoundError:     
            pair_nodes = reconstruction_variables_bdgm(basins, s_regions, dz, alpha, beta, ws_regions, outpath, model)
        
        ensembles_generation(N, pair_nodes, outpath, model)
        
    elif (model == 'model_K' or model == 'model_S' or model == 'model_D'):
    
        if (model == 'model_K'):
            alpha = 1
            beta = 0
            
        elif (model == 'model_D'):
            
            alpha = 1
            beta = 1
            
        elif (model == 'model_S'):
            
            alpha = 0.80
            beta = 0
        
        else:
            
            print('Wrong parameters chosen for model!')
        
        L = len(el)*(0.109/0.067)

        try:
            dz = pd.read_csv(outpath / 'z_{}.csv'.format(model), index_col = None, header = None)
            z = dz[0].item()
        except FileNotFoundError:    
            z = z_computation(basins, outpath, z0, alpha, beta, model, L)   
            
        try:
            pair_nodes = pd.read_csv(outpath / 'rec_variables_{}.csv'.format(model), index_col = 0)
        except FileNotFoundError:     
            pair_nodes = reconstruction_variables(basins, z, alpha, beta, outpath, model)
        
        ensembles_generation(N, pair_nodes, outpath, model)