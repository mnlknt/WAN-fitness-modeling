"""
Metapopulation SIR epidemics on the REAL (WAN / basin) network.

This script only simulates the epidemic on the real network. It is independent
of any reconstruction model. The reconstructed-network simulations live in
`metapopulation_simulations_on_recnet.py`, and the statistical comparison lives in the
separate analysis scripts.

Output:
  METASIMULATIONS/<data_type>/opendata/real/
      real_graph_SIR_<data_type>_seed<seed>_sim<NNN>.pkl
  one pickle per (hub, realization).
"""

import os
import random
import argparse

import numpy as np
import pandas as pd
import pickle

import pycountry_convert as pc
from epidemik import MetaEpiModel


# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DEFAULT_DATA_TYPE = 'basin-WAN'
DEFAULT_BETA_RATE = 0.3
DEFAULT_MU_RATE = 0.1
DEFAULT_T_MAX = 300
DEFAULT_I0 = 10
DEFAULT_N_SIMS = 10        # stochastic realizations per hub on the real network

# Johannesburg, CapeTown, Casablanca, Tunis, HongKong,
# Bangkok, Seoul, Tokyo, London, Paris,
# Milan, Barcelona, New York, Los Angeles, San Francisco,
# Miami, Sydney, Melbourne, Brisbane, Aukland,
# Sao Paulo, Rio de Janeiro, Brasilia, Buenos Aires
DEFAULT_HUBS = [626, 635, 814, 1790, 473,
                538, 2803, 3106, 791, 1535,
                3093, 2884, 2518, 2393, 2375,
                2305, 1612, 1648, 1603, 3046,
                2042, 1990, 1961, 3205]


# ----------------------------------------------------------------------------
# Network / node helpers
# ----------------------------------------------------------------------------
def get_continent(nodes):
    continents = []
    for state in nodes['Country']:
        if state == 'KOS' or state == 'CYN':
            continents.append('EU')
        elif state == 'SXM':
            continents.append('NA')
        elif state == 'TLS':
            continents.append('AS')
        else:
            continents.append(pc.country_alpha2_to_continent_code(pc.country_alpha3_to_country_alpha2(state)))
    return continents


def load_real_network():
    # Import WAN network (basin database)
    network = pd.read_csv('IN_DATA/basin_el_final.csv',
                          usecols=['basin_id1', 'basin_id2', 'weight'],
                          dtype={'weight': 'float'})
    network.rename(columns={'basin_id1': 'Origin',
                            'basin_id2': 'Destination',
                            'weight': 'Passengers'},
                   inplace=True)

    reversed_network = network.rename(columns={'Origin': 'Destination', 'Destination': 'Origin'})
    network = pd.concat([network, reversed_network], ignore_index=True).sort_values(by='Origin').reset_index(drop=True)

    # Import population data
    nodes = pd.read_csv('IN_DATA/basin_info.csv', usecols=['basin_id', 'basin_label', 'country_iso3', 'basin_population'])
    nodes.rename(columns={'basin_id': 'Basin', 'basin_label': 'City', 'country_iso3': 'Country', 'basin_population': 'Population'}, inplace=True)
    nodes['Continent'] = get_continent(nodes)
    nodes = nodes[nodes.index.isin(network['Origin'].unique())]

    network = network.merge(nodes[['Population']], left_on='Origin', right_index=True).rename(columns={'Population': 'Population_Origin'})
    network = network.merge(nodes[['Population']], left_on='Destination', right_index=True).rename(columns={'Population': 'Population_Destination'})
    network['Weight'] = network['Passengers'].div(network['Population_Origin'] + network['Population_Destination'])

    travel = pd.pivot_table(network,
                            index='Origin',
                            columns='Destination',
                            values='Weight',
                            aggfunc='sum')

    travel = travel.fillna(0.) + np.diag(1. - travel.sum(axis=1))

    return nodes, network, travel


def find_hubs(nodes, network):
    nodes = nodes.copy()
    nodes['Strengths'] = 0.0

    strengths = network.groupby('Origin')['Passengers'].sum()
    nodes = nodes.set_index('Basin')
    nodes['Strengths'] = strengths.reindex(nodes.index, fill_value=0.0)
    nodes = nodes.reset_index()

    hubs = []
    for continent in np.unique(nodes['Continent']):
        temp = nodes.loc[nodes['Continent'] == continent]
        temp = temp.sort_values(by='Strengths', ascending=False)
        hubs.extend(temp['Basin'][:4])

    return hubs


def describe_seed(real_nodes, seed):
    """Return (city, country, continent) for a seed basin id from the real network."""
    row = real_nodes.loc[real_nodes['Basin'] == seed]
    if len(row) == 0:
        return f'basin {seed}', '?', '?'
    return (np.array(row['City'])[-1],
            np.array(row['Country'])[-1],
            np.array(row['Continent'])[-1])


def run_and_save(travel, node_table, beta_rate, mu_rate, t_max, seed, I0, file_out):
    """Build a fresh SIR metapopulation model, simulate, and pickle the result."""
    model = MetaEpiModel(travel, node_table)
    model.add_interaction('S', 'I', 'I', rate=beta_rate)
    model.add_spontaneous('I', 'R', rate=mu_rate)
    model.simulate(t_max, seed_state=seed, I=I0)
    with open(file_out, 'wb') as fp:
        pickle.dump(model.models, fp)
    del model


# ----------------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Metapopulation SIR epidemics on the REAL (WAN) network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Example:\n'
               '  python %(prog)s --beta-rate 0.4 --t-max 200 --n-sims 10')

    parser.add_argument('--data-type', default=DEFAULT_DATA_TYPE,
                        help='Database / data type label.')
    parser.add_argument('--beta-rate', type=float, default=DEFAULT_BETA_RATE,
                        help='SIR transmission rate (S + I -> I).')
    parser.add_argument('--mu-rate', type=float, default=DEFAULT_MU_RATE,
                        help='SIR recovery rate (I -> R).')
    parser.add_argument('--t-max', type=int, default=DEFAULT_T_MAX,
                        help='Maximum simulation time.')
    parser.add_argument('--I0', '--i0', dest='I0', type=int, default=DEFAULT_I0,
                        help='Number of infected individuals seeded at t=0.')
    parser.add_argument('--hubs', type=int, nargs='+', default=None,
                        help='Basin ids used as epidemic seeds '
                             '(default: the built-in hub list).')
    parser.add_argument('--find-hubs', action='store_true',
                        help='Compute the hubs from the real network instead of using --hubs.')
    parser.add_argument('--n-sims', type=int, default=DEFAULT_N_SIMS,
                        help='Number of stochastic realizations per hub on the real network.')

    return parser.parse_args()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()

    data_type = args.data_type
    beta_rate = args.beta_rate
    mu_rate = args.mu_rate
    t_max = args.t_max
    I0 = args.I0
    n_sims = args.n_sims

    print('#########################################', flush=True)
    print('METAPOPULATION EPIDEMICS - REAL NETWORK', flush=True)
    print(f'Database: {data_type}', flush=True)
    print('Compartmental model: SIR', flush=True)
    print(f'Transmission rate: {beta_rate}', flush=True)
    print(f'Recovery rate: {mu_rate}', flush=True)
    print(f'Max time: {t_max}', flush=True)
    print(f'Infected at time 0: {I0}', flush=True)
    print(f'Simulations per hub: {n_sims}', flush=True)
    print('-----------------------------------------', flush=True)

    print('Loading real network..', flush=True)
    real_nodes, real_network, real_travel = load_real_network()
    all_population = np.sum(real_nodes['Population'])
    print('Real network loaded.', flush=True)
    print(f'Number of nodes with s>0 (in the real network): {len(real_nodes)}', flush=True)
    print(f'Total population: {all_population}', flush=True)
    print('-----------------------------------------', flush=True)

    print('Determining the hubs: simulations will start from there..', flush=True)
    if args.find_hubs:
        if args.hubs is not None:
            print('Note: --find-hubs given, ignoring the explicit --hubs list.', flush=True)
        hubs = find_hubs(real_nodes, real_network)
    elif args.hubs is not None:
        hubs = args.hubs
    else:
        hubs = DEFAULT_HUBS
    print(f'The hubs: {hubs}', flush=True)
    print('-----------------------------------------', flush=True)

    print('===== Simulations on the REAL network =====', flush=True)
    real_dir = f'METASIMULATIONS/{data_type}/opendata/real/'
    os.makedirs(real_dir, exist_ok=True)
    print(f'Real-network output directory ready: {real_dir}', flush=True)
    real_node_table = real_nodes[['Basin', 'Population']]
    for seed in hubs:
        city, country, continent = describe_seed(real_nodes, seed)
        print(f'--> Epidemics breaks in {city} ({country}, {continent})', flush=True)
        for sim in range(n_sims):
            file_out = f'{real_dir}real_graph_SIR_{data_type}_seed{seed}_sim{sim+1:03d}.pkl'
            print(f'    real network, seed {seed}, sim {sim + 1}/{n_sims}..', flush=True)
            run_and_save(real_travel, real_node_table,
                         beta_rate, mu_rate, t_max, seed, I0, file_out)
    print('Real-network simulations finished.', flush=True)
    print('#########################################', flush=True)


if __name__ == "__main__":
    main()