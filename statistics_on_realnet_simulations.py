"""
Statistical analysis of the metapopulation SIR simulations on the REAL network.

The real-network epidemic is simulated as several stochastic REALIZATIONS
(sim000, sim001, ...) by `simulate_real_network_opendata.py`. For each hub this
script compares all pairs of realizations, giving the baseline (noise-floor)
spread of the dynamics on the real network. It uses the same metric suite as the
shared analysis scripts: area between global-prevalence curves, L2 distance,
peak-time / peak-magnitude / epidemic-length differences, per-basin prevalence
RMSE, and entropy L2.

Input:
  METASIMULATIONS/<data_type>/opendata/real/
      real_graph_SIR_<data_type>_seed<seed>_sim<NNN>.pkl
"""

import os
import re
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

EPS = sys.float_info.epsilon
epidemic_threshold = 1e-4

DEFAULT_DATA_TYPE = 'basin-WAN'
DEFAULT_T_MAX = 300

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
# Metric helpers (shared with the other analysis scripts)
# ----------------------------------------------------------------------------
def compute_area_between_curves(x, y, dt=1.0):
    return np.sum(np.abs(np.asarray(x) - np.asarray(y))) * dt


def compute_l2_distance(x, y):
    return np.sqrt(np.sum((np.asarray(x) - np.asarray(y)) ** 2))


def compute_peak_time(curve, smooth=True):
    curve = np.asarray(curve)
    if smooth:
        curve = smooth_curve(curve)
    return int(np.argmax(curve))


def compute_peak_magnitude(curve, smooth=True):
    curve = np.asarray(curve)
    if smooth:
        curve = smooth_curve(curve)
    return float(np.max(curve))


def compute_epidemic_length(curve, threshold=epidemic_threshold):
    curve = np.asarray(curve)
    idx = np.argwhere(curve > threshold)
    if idx.size == 0:
        return 0
    return int(idx[-1, 0] - idx[0, 0] + 1)


def smooth_curve(curve, window=11, polyorder=2):
    """Savitzky-Golay smoothing with safety checks."""
    curve = np.asarray(curve)
    if window >= len(curve):
        window = len(curve) - 1 if len(curve) % 2 == 0 else len(curve)
    if window % 2 == 0:
        window += 1
    return savgol_filter(curve, window_length=window, polyorder=polyorder)


def load_prevalence(sim, nodes, t_max):
    """
    Return:
      prevalence : (n_nodes, t_max+1)  per-basin I/(S+I+R), capped in [0, 1].
      infected   : (t_max+1,)          global sum(I)/sum(S+I+R) per time step.

    Uses the time-evolving simulator population N = S+I+R rather than the
    static CSV population, so transit-heavy basins (small residents, many
    visitors) don't produce values above 1.
    """
    prevalence = np.zeros((len(nodes), t_max + 1))
    sum_I = np.zeros(t_max + 1)
    sum_N = np.zeros(t_max + 1)

    for node in nodes['Basin']:
        if node not in sim:
            continue
        S = np.asarray(sim[node]['S'], dtype=float)
        I = np.asarray(sim[node]['I'], dtype=float)
        R = np.asarray(sim[node]['R'], dtype=float)
        N = S + I + R
        with np.errstate(divide='ignore', invalid='ignore'):
            prevalence[node] = np.where(N > 0, I / N, 0.0)
        sum_I += I
        sum_N += N

    with np.errstate(divide='ignore', invalid='ignore'):
        infected = np.where(sum_N > 0, sum_I / sum_N, 0.0)
    return prevalence, infected


def entropy_of(prevalence, n_nodes):
    rho = prevalence / (np.sum(prevalence, axis=0, keepdims=True) + EPS)
    rho = np.clip(rho, a_min=EPS, a_max=1.0)
    return -np.sum(rho * np.log(rho), axis=0) / np.log(n_nodes)


def compute_pair_metrics(a, b):
    """a, b = (prevalence, infected, entropy). Return a dict of difference metrics."""
    a_prev, a_inf, a_ent = a
    b_prev, b_inf, b_ent = b
    RMSE = np.sqrt(np.mean((a_prev - b_prev) ** 2, axis=0))
    L2 = np.abs(a_ent - b_ent)
    return {
        'area_between_curves': compute_area_between_curves(a_inf, b_inf),
        'l2_distance': compute_l2_distance(a_inf, b_inf),
        'peak_time_difference': np.abs(compute_peak_time(a_inf) - compute_peak_time(b_inf)),
        'peak_magnitude_difference': np.abs(compute_peak_magnitude(a_inf) - compute_peak_magnitude(b_inf)),
        'epidemic_length_difference': np.abs(compute_epidemic_length(a_inf) - compute_epidemic_length(b_inf)),
        'rmse_prevalence': np.mean(RMSE),
        'l2_entropy': np.sum(L2),
    }


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def list_real_files(data_type, hub):
    """Real-network realization pickles for a given hub, sorted by sim index."""
    d = f'METASIMULATIONS/{data_type}/opendata/real'
    if not os.path.isdir(d):
        return []
    pat = re.compile(rf'^real_graph_SIR_{re.escape(data_type)}_seed{hub}_sim\d+\.pkl$')
    return sorted(os.path.join(d, f) for f in os.listdir(d) if pat.match(f))


def load_sim_summary(path, nodes, t_max):
    with open(path, 'rb') as fp:
        sim = pickle.load(fp)
    prev, inf = load_prevalence(sim, nodes, t_max)
    ent = entropy_of(prev, len(nodes))
    return prev, inf, ent


def load_nodes():
    nodes = pd.read_csv('IN_DATA/basin_info.csv',
                        usecols=['basin_id', 'basin_label', 'country_iso3', 'basin_population'])
    nodes.rename(columns={'basin_id': 'Basin', 'basin_label': 'City',
                          'country_iso3': 'Country', 'basin_population': 'Population'}, inplace=True)
    return nodes


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze the SIR simulations on the REAL network '
                    '(pairwise comparison of stochastic realizations).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-type', default=DEFAULT_DATA_TYPE,
                        help='Database / data type label.')
    parser.add_argument('--t-max', type=int, default=DEFAULT_T_MAX,
                        help='Simulation length (must match the simulations).')
    parser.add_argument('--hubs', type=int, nargs='+', default=DEFAULT_HUBS,
                        help='Basin ids that were used as epidemic seeds.')
    return parser.parse_args()


def main():
    args = parse_args()
    data_type = args.data_type
    t_max = args.t_max
    hubs = args.hubs

    print('--- Load real network data', flush=True)
    nodes = load_nodes()

    METRIC_KEYS = ['area_between_curves', 'l2_distance', 'peak_time_difference',
                   'peak_magnitude_difference', 'epidemic_length_difference',
                   'rmse_prevalence', 'l2_entropy']
    hub_metrics = {k: [] for k in METRIC_KEYS}
    real_prevalence = {}

    for hub in hubs:
        files = list_real_files(data_type, hub)
        if not files:
            print(f'+++ Warning: hub {hub} has no real realizations. Skipping.', flush=True)
            continue

        # Load every realization for this hub once.
        sims = [load_sim_summary(f, nodes, t_max) for f in files]
        real_prevalence[hub] = np.mean([s[0] for s in sims], axis=0)

        if len(files) < 2:
            print(f'+++ Warning: hub {hub} has {len(files)} real realization(s); '
                  f'need >= 2 to compare. Skipping pairwise metrics.', flush=True)
            continue

        # Collect all pairs for this hub, then store the per-hub mean.
        # Averaging within hub before averaging across hubs gives equal weight
        # to each hub regardless of how many realizations it has.
        hub_pairs = {k: [] for k in METRIC_KEYS}
        for i in range(len(sims)):
            for j in range(i):
                m = compute_pair_metrics(sims[i], sims[j])
                for k in METRIC_KEYS:
                    hub_pairs[k].append(m[k])
        for k in METRIC_KEYS:
            hub_metrics[k].append(np.mean(hub_pairs[k]))

        print(f'--- Hub {hub}: {len(files)} realizations, '
              f'{len(files) * (len(files) - 1) // 2} pairs.', flush=True)

    if real_prevalence:
        out_path = f'METASIMULATIONS/{data_type}/opendata/real_prevalence.pkl'
        with open(out_path, 'wb') as fp:
            pickle.dump(real_prevalence, fp)
        print(f'--- Real prevalence saved to {out_path}', flush=True)

    print('### FINAL AVERAGED METRICS OVER ALL SEEDS (real-net realization pairs) ###', flush=True)
    if not hub_metrics['area_between_curves']:
        print('No comparisons were made (no hub had >= 2 realizations).', flush=True)
        return
    print(f"Area between curves: {np.mean(hub_metrics['area_between_curves'])} +- {np.std(hub_metrics['area_between_curves'])}", flush=True)
    print(f"L2 distance: {np.mean(hub_metrics['l2_distance'])} +- {np.std(hub_metrics['l2_distance'])}", flush=True)
    print(f"Peak time difference: {np.mean(hub_metrics['peak_time_difference'])} +- {np.std(hub_metrics['peak_time_difference'])}", flush=True)
    print(f"Peak magnitude difference: {np.mean(hub_metrics['peak_magnitude_difference'])} +- {np.std(hub_metrics['peak_magnitude_difference'])}", flush=True)
    print(f"Epidemic length difference: {np.mean(hub_metrics['epidemic_length_difference'])} +- {np.std(hub_metrics['epidemic_length_difference'])}", flush=True)
    print(f"RMSE prevalence: {np.mean(hub_metrics['rmse_prevalence'])} +- {np.std(hub_metrics['rmse_prevalence'])}", flush=True)
    print(f"L2 entropy: {np.mean(hub_metrics['l2_entropy'])} +- {np.std(hub_metrics['l2_entropy'])}", flush=True)
    print('--------------------------------------------------------------', flush=True)


if __name__ == "__main__":
    main()