"""
Combined plotting for the metapopulation SIR simulations. For every hub it
produces BOTH:

  1. S / I / R vs time curves
     - real network: mean +/- std band (red) across the real-net realizations
       (model-independent);
     - reconstructed networks: one blue curve per network in `recList`.

  2. Epidemics-on-map snapshot at time `this_time`
     - real network and the network(s) in `map_recList`, drawn on a world map,
       coloured by per-basin prevalence.

All figures are written to a top-level FIGURES/ directory (sibling of
METASIMULATIONS/).

Usage:
  python plot_metapopulation_simulations.py --data_type basin-WAN --model model_B
"""

import sys
import os
import re
import pickle
import argparse

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use('Agg')  # file-only output; safe for headless / batch runs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import seaborn as sns

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Use a local copy of the Natural Earth shapefiles if it sits next to this
# script (folder 'cartopy_data/'). This avoids cartopy trying to download them
# from the network at render time, which fails on restricted connections.
_local_ne = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cartopy_data')
if os.path.isdir(_local_ne):
    cartopy.config['pre_existing_data_dir'] = _local_ne


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
t_max = 300

hubs = [626, 635, 814, 1790, 473,       # Johannesburg, CapeTown, Casablanca, Tunis, HongKong
        538, 2803, 3106, 791, 1535,     # Bangkok, Seoul, Tokyo, London, Paris,
        3093, 2884, 2518, 2393, 2375,   # Milan, Barcelona, New York, Los Angeles, San Francisco
        2305, 1612, 1648, 1603, 3046,   # Miami, Sydney, Melbourne, Brisbane, Aukland
        2042, 1990, 1961, 3205]         # São Paulo, Rio de Janeiro, Brasilia, Buenos Aires

# Reconstructed networks drawn as curves in the vs-time plots.
recList = [9, 97, 24, 16, 54, 8, 36, 66, 84, 30, 74, 82, 18, 63, 90, 53, 2, 60, 69, 80]

# Reconstructed network(s) rendered on the map (one snapshot each).
map_recList = [69]

my_red = (202. / 255, 0., 32. / 255)
my_blue = (5. / 255, 113. / 255, 176. / 255)

# All figures go here: a top-level FIGURES/ directory (sibling of METASIMULATIONS).
FIGURES_DIR = 'FIGURES'

# Seaborn rocket palette (reversed) for the map colouring.
rocket_cmap = sns.color_palette("rocket_r", as_cmap=True)


# ----------------------------------------------------------------------------
# Map rendering (from plot_epidemics_on_map_opendata.py)
# ----------------------------------------------------------------------------
def display_epidemics(G, population_dict, hub_id, filename):
    # Filter nodes with prevalence greater than or equal to 10^(-3)
    filtered_nodes = [n for n in G.nodes if G.nodes[n]['prevalence'] >= 1e-3]

    # Create subgraph for filtered nodes
    filtered_G = G.subgraph(filtered_nodes)

    # Plot settings: a PlateCarree map whose data coordinates ARE lon/lat,
    # matching the basin positions.
    fig = plt.figure(figsize=(24, 10), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    # World map underneath the data: light-blue ocean + light-gray landmasses.
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#d9d9d9', edgecolor='none', zorder=1)
    ax.coastlines(linewidth=0.3, color='white', zorder=2)

    # Node coordinates, colors and sizes.
    lons = [G.nodes[n]['longitude'] for n in filtered_G.nodes]
    lats = [G.nodes[n]['latitude'] for n in filtered_G.nodes]
    node_colors = [G.nodes[n]['prevalence'] for n in filtered_G.nodes]

    # Apply logarithmic normalization to prevalence values
    norm = LogNorm(vmin=1e-3, vmax=1)
    node_colors_normalized = [rocket_cmap(norm(value)) for value in node_colors]

    # Node sizes from population on a log2 scale for more differentiation
    node_sizes = [max(5, np.log2(population_dict[n]) * 15) for n in filtered_G.nodes]

    # Plot the basins on top of the map.
    ax.scatter(lons, lats, s=node_sizes, color=node_colors_normalized,
               alpha=0.6, transform=ccrs.PlateCarree(), zorder=5)

    # Add a color bar for prevalence with logarithmic scale
    sm = plt.cm.ScalarMappable(cmap=rocket_cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=40)
    cbar.set_label('Prevalence', fontsize=48)

    # Highlight the hub node with a star symbol
    ax.scatter(G.nodes[hub_id]['longitude'], G.nodes[hub_id]['latitude'],
               s=3000, color='red', marker='*', edgecolors='black',
               linewidth=2.5, zorder=6, transform=ccrs.PlateCarree(), label='Hub')

    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#e0f7fa')
    plt.close()


# ----------------------------------------------------------------------------
# Per-hub vs-time plots (from plot_metapopulation_simulation_vs_time_opendata.py)
# ----------------------------------------------------------------------------
def plot_vs_time(data_type, model, nodes, hub):
    # Real curve = mean +/- std band across the real-net REALIZATIONS
    # (model-independent), using only S/I/R.
    real_dir = f'METASIMULATIONS/{data_type}/opendata/real'
    if os.path.isdir(real_dir):
        pat = re.compile(rf'^real_graph_SIR_{re.escape(data_type)}_seed{hub}_sim\d+\.pkl$')
        real_files = sorted(os.path.join(real_dir, f) for f in os.listdir(real_dir) if pat.match(f))
    else:
        real_files = []

    S_runs, I_runs, R_runs = [], [], []
    for rf in real_files:
        with open(rf, 'rb') as fp:
            real_model = pickle.load(fp)

        sum_S = np.zeros(t_max + 1)
        sum_I = np.zeros(t_max + 1)
        sum_R = np.zeros(t_max + 1)
        for node in nodes['Basin']:
            if node in real_model:
                sum_S += np.asarray(real_model[node]['S'], dtype=float)
                sum_I += np.asarray(real_model[node]['I'], dtype=float)
                sum_R += np.asarray(real_model[node]['R'], dtype=float)
        sum_N = sum_S + sum_I + sum_R
        with np.errstate(divide='ignore', invalid='ignore'):
            S_runs.append(np.where(sum_N > 0, sum_S / sum_N, 0.0))
            I_runs.append(np.where(sum_N > 0, sum_I / sum_N, 0.0))
            R_runs.append(np.where(sum_N > 0, sum_R / sum_N, 0.0))

    have_real = len(S_runs) > 0
    if have_real:
        S_runs = np.array(S_runs)
        I_runs = np.array(I_runs)
        R_runs = np.array(R_runs)
        real_S, real_I, real_R = S_runs.mean(axis=0), I_runs.mean(axis=0), R_runs.mean(axis=0)
        up_S, low_S = real_S + S_runs.std(axis=0), real_S - S_runs.std(axis=0)
        up_I, low_I = real_I + I_runs.std(axis=0), real_I - I_runs.std(axis=0)
        up_R, low_R = real_R + R_runs.std(axis=0), real_R - R_runs.std(axis=0)
    else:
        print(f'+++ Warning: no real-net realizations for hub {hub}; plotting rec curves only.', flush=True)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    ax1.set_ylabel('Susceptible')
    ax2.set_ylabel('Infected')
    ax3.set_ylabel('Recovered')
    ax3.set_xlabel('Time')
    ax1.set_xlim(0, 150)
    ax2.set_xlim(0, 150)
    ax3.set_xlim(0, 150)

    for rec_net in recList:
        print(f'Retrieve metapopulation epidemics result on rec network {rec_net}. Seed {hub}.', flush=True)
        in_file = (f'METASIMULATIONS/{data_type}/opendata/{model}/'
                   f'rec_graph_SIR_{data_type}_{model}_net{rec_net - 1:04d}_seed{hub}.pkl')
        with open(in_file, 'rb') as fp:
            rec_model = pickle.load(fp)

        rec_S = np.zeros(t_max + 1)
        rec_I = np.zeros(t_max + 1)
        rec_R = np.zeros(t_max + 1)

        tot_population = 0
        for node in nodes['Basin']:
            if node in rec_model:
                tot_population += (rec_model[node]['S'] + rec_model[node]['I'] + rec_model[node]['R'])
                rec_S += np.array(rec_model[node]['S'])
                rec_I += np.array(rec_model[node]['I'])
                rec_R += np.array(rec_model[node]['R'])

        rec_S /= tot_population
        rec_I /= tot_population
        rec_R /= tot_population

        ax1.plot(rec_S, color=my_blue, alpha=0.5)
        ax2.plot(rec_I, color=my_blue, alpha=0.5)
        ax3.plot(rec_R, color=my_blue, alpha=0.5)

    if have_real:
        ax1.fill_between(np.arange(t_max + 1), low_S, up_S, color=my_red, alpha=0.3)
        ax1.plot(real_S, color=my_red)
        ax2.fill_between(np.arange(t_max + 1), low_I, up_I, color=my_red, alpha=0.3)
        ax2.plot(real_I, color=my_red)
        ax3.fill_between(np.arange(t_max + 1), low_R, up_R, color=my_red, alpha=0.3)
        ax3.plot(real_R, color=my_red)

    custom_handles = [
        Line2D([0], [0], color=my_red, label='real net'),
        Line2D([0], [0], color=my_blue, label='rec net'),
    ]
    ax3.legend(handles=custom_handles, loc='best')
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fig, comp in ((fig1, 'S'), (fig2, 'I'), (fig3, 'R')):
        fig_file = f'{FIGURES_DIR}/metapopulation_simulation_vs_time_seed{hub}_{model}_{comp}.png'
        if os.path.exists(fig_file):
            os.remove(fig_file)
        fig.savefig(fig_file)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Per-hub map plots (from plot_epidemics_on_map_opendata.py)
# ----------------------------------------------------------------------------
def plot_maps(data_type, model, real_prevalence, rec_prevalence, real_G, rec_G, basin_ids, population_dict, hub, this_time):
    out_dir = FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)

    for node_id in basin_ids:
        try:
            real_G.nodes[node_id]['prevalence'] = real_prevalence[hub][node_id][this_time]
        except (KeyError, IndexError):
            print(f"Prevalence data for node {node_id} or hub {hub} not found.")
            real_G.nodes[node_id]['prevalence'] = 0

    filename = f'{out_dir}/epidemics_on_map_{model}_realnet_seed{hub}.png'
    display_epidemics(real_G, population_dict, hub, filename)
    print(f'Seed {hub} real net.', flush=True)

    for rec_net in map_recList:
        for node_id in basin_ids:
            try:
                rec_G.nodes[node_id]['prevalence'] = rec_prevalence[hub][rec_net][node_id][this_time]
            except (KeyError, IndexError):
                print(f"Prevalence data for node {node_id} or hub {hub} not found.")
                rec_G.nodes[node_id]['prevalence'] = 0

        filename = f'{out_dir}/epidemics_on_map_{model}_recnet{rec_net - 1}_seed{hub}.png'
        display_epidemics(rec_G, population_dict, hub, filename)
        print(f'Seed {hub} rec net {rec_net}.', flush=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Plot metapopulation simulations: vs-time curves and epidemics-on-map.')
    parser.add_argument('--data_type', type=str, default='basin-WAN', help='Data type (e.g., basin-WAN)')
    parser.add_argument('--model', type=str, required=True, help='Reconstruction model (e.g., model_B)')
    parser.add_argument('--hub', type=str, default='London(UK)', help='City name of the epidemic seed (case-insensitive, matched against basin_label in basin_info.csv).')
    parser.add_argument('--time', type=int, default=80, help='Time step at which to snapshot the map.')
    args = parser.parse_args()

    data_type = args.data_type
    model = args.model
    this_time = args.time

    print('######################################', flush=True)
    print('PLOTTING METAPOPULATION SIMULATIONS', flush=True)
    print(f'Dataset: {data_type}', flush=True)
    print(f'Reconstruction model: {model}', flush=True)
    print('----------------------------------------------------------------', flush=True)

    # --- basin info, in both the forms the two original scripts used ---
    basin_info = pd.read_csv('IN_DATA/basin_info.csv', sep=',')

    match = basin_info[basin_info['basin_label'].str.lower() == args.hub.lower()]
    if match.empty:
        print(f'Error: city "{args.hub}" not found in basin_info.csv. '
              f'Available cities include: {", ".join(basin_info["basin_label"].head(10).tolist())} ...', flush=True)
        return
    if len(match) > 1:
        print(f'Warning: "{args.hub}" matched multiple basins {match["basin_id"].tolist()}; using the first.', flush=True)
    hub = int(match.iloc[0]['basin_id'])
    print(f'Hub resolved: {args.hub!r} -> basin_id {hub}', flush=True)

    nodes = basin_info[['basin_id', 'basin_label', 'country_iso3', 'basin_population']].copy()
    nodes.rename(columns={'basin_id': 'Basin', 'basin_label': 'City',
                          'country_iso3': 'Country', 'basin_population': 'Population'}, inplace=True)

    coords = basin_info[['latitude', 'longitude']].to_numpy()
    basin_ids = basin_info['basin_id'].to_numpy()
    basin_populations = basin_info['basin_population'].to_numpy()
    population_dict = {basin_ids[i]: basin_populations[i] for i in range(len(basin_ids))}

    # --- prevalence snapshot dicts used by the map plots ---
    with open(f'METASIMULATIONS/{data_type}/opendata/real_prevalence.pkl', 'rb') as fp:
        real_prevalence = pickle.load(fp)
    with open(f'METASIMULATIONS/{data_type}/opendata/rec_prevalence_{model}.pkl', 'rb') as fp:
        rec_prevalence = pickle.load(fp)

    # --- node graphs (coordinates) reused for every map ---
    real_G = nx.Graph()
    rec_G = nx.Graph()
    for i, node_id in enumerate(basin_ids):
        for G in (real_G, rec_G):
            G.add_node(node_id)
            G.nodes[node_id]['latitude'] = coords[i][0]
            G.nodes[node_id]['longitude'] = coords[i][1]
    print('Nodes in network: ', len(real_G.nodes), flush=True)

    print(f'Retrieve metapopulation epidemics result on real network. Seed {hub}.', flush=True)
    plot_vs_time(data_type, model, nodes, hub)
    plot_maps(data_type, model, real_prevalence, rec_prevalence, real_G, rec_G, basin_ids, population_dict, hub, this_time)

    print('######################################', flush=True)
    print('######################################', flush=True)
    print('######################################', flush=True)


if __name__ == "__main__":
    main()