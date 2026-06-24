"""
Metapopulation SIR epidemics on the RECONSTRUCTED (WAN / basin) networks.

This script only simulates the epidemic on the reconstructed networks produced
by one or more reconstruction models. The real-network simulations live in
`metapopulation_simulation_on_realnet.py`, and the statistical comparison lives in the
separate analysis scripts.

Reconstructed samples are read from:
  <samples_dir>/<model>/edgelist_<k>.csv     (default <samples_dir> = Ensembles)

Output:
  METASIMULATIONS/<data_type>/opendata/<model>/
      rec_graph_SIR_<data_type>_<model>_net<NNNN>_seed<seed>[_sim<NNN>].pkl
  one pickle per (reconstructed network, hub, realization).
"""

import os
import re
import glob
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
DEFAULT_SAMPLES_DIR = 'Ensembles'   # generator writes <savepath>Ensembles/<model>/edgelist_<i>.csv
DEFAULT_BETA_RATE = 0.3
DEFAULT_MU_RATE = 0.1
DEFAULT_T_MAX = 300
DEFAULT_I0 = 10
DEFAULT_N_SIMS = 1          # stochastic realizations per hub on each reconstructed network
DEFAULT_N_REC = 20          # number of reconstructed networks to draw at random

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

# Default reconstruction model (the sub-folder expected under --samples-dir).
# Edit this to match your own reconstruction-model name.
DEFAULT_MODEL = 'model_A'

# Default reconstructed-network list (the 20 networks originally hard-coded).
# Used unless --rec-list is given, or --random-rec is set to draw at random.
DEFAULT_REC_LIST = [9, 97, 24, 16, 54, 8, 36, 66, 84, 30,
                    74, 82, 18, 63, 90, 53, 2, 60, 69, 80]


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


def load_real_nodes():
    """Light real-network node table (no travel matrix), used for seed labels."""
    nodes = pd.read_csv('IN_DATA/basin_info.csv',
                        usecols=['basin_id', 'basin_label', 'country_iso3', 'basin_population'])
    nodes.rename(columns={'basin_id': 'Basin', 'basin_label': 'City',
                          'country_iso3': 'Country', 'basin_population': 'Population'}, inplace=True)
    nodes['Continent'] = get_continent(nodes)
    return nodes


def load_real_network():
    """Full real network (nodes + edges + travel matrix); needed only for --find-hubs."""
    network = pd.read_csv('IN_DATA/basin_el_final.csv',
                          usecols=['basin_id1', 'basin_id2', 'weight'],
                          dtype={'weight': 'float'})
    network.rename(columns={'basin_id1': 'Origin',
                            'basin_id2': 'Destination',
                            'weight': 'Passengers'},
                   inplace=True)

    reversed_network = network.rename(columns={'Origin': 'Destination', 'Destination': 'Origin'})
    network = pd.concat([network, reversed_network], ignore_index=True).sort_values(by='Origin').reset_index(drop=True)

    nodes = load_real_nodes()
    nodes = nodes[nodes.index.isin(network['Origin'].unique())]

    network = network.merge(nodes[['Population']], left_on='Origin', right_index=True).rename(columns={'Population': 'Population_Origin'})
    network = network.merge(nodes[['Population']], left_on='Destination', right_index=True).rename(columns={'Population': 'Population_Destination'})
    network['Weight'] = network['Passengers'].div(network['Population_Origin'] + network['Population_Destination'])

    travel = pd.pivot_table(network, index='Origin', columns='Destination',
                            values='Weight', aggfunc='sum')
    travel = travel.fillna(0.) + np.diag(1. - travel.sum(axis=1))

    return nodes, network, travel


def load_rec_network(rec_net, samples_dir, model):
    network = pd.read_csv(os.path.join(samples_dir, model, 'edgelist_%d.csv' % (rec_net - 1)),
                          sep=',', names=['Origin', 'Destination', 'Passengers'],
                          index_col=None, comment='#', skiprows=1,
                          dtype={'Passengers': 'float'})

    reversed_network = network.rename(columns={'Origin': 'Destination', 'Destination': 'Origin'})
    network = pd.concat([network, reversed_network], ignore_index=True).sort_values(by='Origin').reset_index(drop=True)

    nodes = pd.read_csv('IN_DATA/basin_info.csv', usecols=['basin_id', 'basin_label', 'country_iso3', 'basin_population'])
    nodes.rename(columns={'basin_id': 'Basin', 'basin_label': 'City', 'country_iso3': 'Country', 'basin_population': 'Population'}, inplace=True)
    nodes = nodes[nodes.index.isin(network['Origin'].unique())]

    network = network.merge(nodes[['Population']], left_on='Origin', right_index=True).rename(columns={'Population': 'Population_Origin'})
    network = network.merge(nodes[['Population']], left_on='Destination', right_index=True).rename(columns={'Population': 'Population_Destination'})
    network['Weight'] = network['Passengers'].div(network['Population_Origin'] + network['Population_Destination'])

    travel = pd.pivot_table(network, index='Origin', columns='Destination',
                            values='Weight', aggfunc='sum')
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


def list_available_rec_networks(samples_dir, model):
    """Return the 1-based ids of the reconstructed networks available on disk.

    Files are stored as <samples_dir>/<model>/edgelist_<k>.csv where the
    0-based file index k corresponds to rec_net = k + 1.
    """
    pattern = os.path.join(samples_dir, model, 'edgelist_*.csv')
    rec_nets = []
    for f in glob.glob(pattern):
        m = re.search(r'edgelist_(\d+)\.csv$', os.path.basename(f))
        if m:
            rec_nets.append(int(m.group(1)) + 1)
    return sorted(rec_nets)


def choose_rec_list(args, model):
    """Decide which reconstructed networks to use for a given model.

    By default the --rec-list value is used (which itself defaults to
    DEFAULT_REC_LIST). If --random-rec is set, draw --n-rec random samples from
    the reconstructed networks available on disk instead.
    """
    if not args.random_rec:
        return list(args.rec_list)

    available = list_available_rec_networks(args.samples_dir, model)
    if not available:
        raise FileNotFoundError(
            "No reconstructed network samples found under "
            f"{os.path.join(args.samples_dir, model)}/ (expected edgelist_*.csv). "
            "Provide them explicitly with --rec-list or set --samples-dir."
        )

    n = min(args.n_rec, len(available))
    if n < args.n_rec:
        print(f'+++ Warning: only {len(available)} reconstructed networks available '
              f'for model {model}, drawing {n} instead of {args.n_rec}.', flush=True)
    return sorted(random.sample(available, n))


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
        description='Metapopulation SIR epidemics on the RECONSTRUCTED (WAN) networks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Example:\n'
               '  python %(prog)s model_A --beta-rate 0.4 --t-max 200 --n-sims 5\n'
               '  python %(prog)s model_A --random-rec --n-rec 20 --rng-seed 42')

    parser.add_argument('rec_model', nargs='?', metavar='REC_MODEL', default=DEFAULT_MODEL,
                        help='The reconstruction model that produced the reconstructed '
                             'samples in <samples_dir>/<model>/ (e.g. model_A). '
                             'Defaults to DEFAULT_MODEL.')

    parser.add_argument('--data-type', default=DEFAULT_DATA_TYPE,
                        help='Database / data type label.')
    parser.add_argument('--samples-dir', default=DEFAULT_SAMPLES_DIR,
                        help='Directory holding the reconstructed samples, with one sub-folder '
                             'per model containing edgelist_<i>.csv files '
                             '(i.e. <samples_dir>/<model>/edgelist_<i>.csv).')
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
                        help='Number of stochastic realizations per hub on each reconstructed network.')

    parser.add_argument('--rec-list', type=int, nargs='+', default=DEFAULT_REC_LIST,
                        help='List of reconstructed networks to simulate (1-based ids). '
                             'Defaults to the built-in DEFAULT_REC_LIST.')
    parser.add_argument('--random-rec', action='store_true',
                        help='Ignore --rec-list and instead draw --n-rec random reconstructed '
                             'networks from disk (per model).')
    parser.add_argument('--n-rec', type=int, default=DEFAULT_N_REC,
                        help='Number of reconstructed networks to draw at random '
                             'when --random-rec is set.')
    parser.add_argument('--rng-seed', type=int, default=None,
                        help='Seed for the RNG used to draw random reconstructed '
                             'networks (for reproducibility).')

    parser.add_argument('--keep-previous-results', dest='delete_previous_results',
                        action='store_false',
                        help='Do not delete the previous .dat results file before starting.')
    parser.set_defaults(delete_previous_results=True)

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

    if args.rng_seed is not None:
        random.seed(args.rng_seed)
        np.random.seed(args.rng_seed)

    print('#########################################', flush=True)
    print('METAPOPULATION EPIDEMICS - RECONSTRUCTED NETWORKS', flush=True)
    print(f'Database: {data_type}', flush=True)
    print(f'Samples directory: {args.samples_dir}', flush=True)
    print(f'Reconstruction model: {args.rec_model}', flush=True)
    print('Compartmental model: SIR', flush=True)
    print(f'Transmission rate: {beta_rate}', flush=True)
    print(f'Recovery rate: {mu_rate}', flush=True)
    print(f'Max time: {t_max}', flush=True)
    print(f'Infected at time 0: {I0}', flush=True)
    print(f'Simulations per hub: {n_sims}', flush=True)
    print('-----------------------------------------', flush=True)

    # Real network is needed only to resolve/print the hubs. Build the full
    # network (with travel matrix) only when hubs must be computed from it.
    print('Determining the hubs: simulations will start from there..', flush=True)
    if args.find_hubs:
        if args.hubs is not None:
            print('Note: --find-hubs given, ignoring the explicit --hubs list.', flush=True)
        real_nodes, real_network, _ = load_real_network()
        hubs = find_hubs(real_nodes, real_network)
    else:
        real_nodes = load_real_nodes()
        hubs = args.hubs if args.hubs is not None else DEFAULT_HUBS
    print(f'The hubs: {hubs}', flush=True)
    print('-----------------------------------------', flush=True)

    model = args.rec_model
    print('=========================================', flush=True)
    print(f'MODEL: {model}', flush=True)
    print('=========================================', flush=True)

    out_dir = f'METASIMULATIONS/{data_type}/opendata/{model}/'
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory ready: {out_dir}', flush=True)

    if args.delete_previous_results:
        filename = f'OUT_DATA/metapopulation_epidemics_{data_type}_opendata_{model}.dat'
        if os.path.exists(filename):
            os.remove(filename)
            print(f"The file {filename} has been deleted to start new simulation program.")
            print('-----------------------------------------', flush=True)

    rec_list = choose_rec_list(args, model)
    print(f'Reconstructed networks for this model: {rec_list}', flush=True)
    print('-----------------------------------------', flush=True)

    print('===== Simulations on the RECONSTRUCTED networks =====', flush=True)
    for rec_net in rec_list:
        print(f'Reconstructed network {rec_net} (file index {rec_net - 1})', flush=True)
        rec_nodes, rec_network, rec_travel = load_rec_network(rec_net, args.samples_dir, model)
        rec_node_table = rec_nodes[['Basin', 'Population']]
        present = set(rec_nodes['Basin'].values)
        print('Rec network loaded.', flush=True)

        for seed in hubs:
            if seed in present:
                for sim in range(n_sims):
                    suffix = '' if n_sims == 1 else f'_sim{sim:03d}'
                    file_out = (f'{out_dir}rec_graph_SIR_{data_type}_{model}'
                                f'_net{rec_net - 1:04d}_seed{seed}{suffix}.pkl')
                    print(f'    rec network {rec_net}, seed {seed}, '
                          f'sim {sim + 1}/{n_sims}..', flush=True)
                    run_and_save(rec_travel, rec_node_table,
                                 beta_rate, mu_rate, t_max, seed, I0, file_out)
            else:
                city, _, _ = describe_seed(real_nodes, seed)
                print(f'+++ Warning: node {city} isolated in rec network {rec_net}', flush=True)
    print('Reconstructed-network simulations finished.', flush=True)
    print('#########################################', flush=True)


if __name__ == "__main__":
    main()