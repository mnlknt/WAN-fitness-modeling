# WAN reconstruction

Code and data for the paper:

> G. Fischetti, A. Mancini, G. Cimini, J.T. Davis, A. Leung, A. Vespignani and G. Caldarelli,  
> *Block-Fitness Modeling of the Global Air Mobility Network* (2026).

This repository contains:

- **Network reconstruction** — fitness-based models that reconstruct the WAN from partial information and write edgelists into `Ensembles/<model>/`.
- **Epidemic simulation and analysis** — runs SIR metapopulation epidemics on both the real network and the reconstructed networks, then compares them to quantify how well each reconstruction model preserves epidemic dynamics.

The reconstruction models and their codes are:

| Code | Article name |
|---|---|
| `dgm` | Model K |
| `nlin-dgm` | Model S |
| `dist-dgm` | Model D |
| `bdgm` | Model B |
| `comb-bdgm` | Model C |

> **Note:** the network reconstruction step must be run first to populate `Ensembles/` before executing step 2 of the epidemic pipeline below.

---

## Directory structure

```
WAN/
├── IN_DATA/
│   ├── basin_info.csv          # basin metadata (id, label, country, population, lat, lon)
│   └── basin_el_final.csv      # real edge list (basin_id1, basin_id2, passengers, distance)
├── Ensembles/                  # user-provided reconstructed network samples
│   └── <model>/
│       └── edgelist_<k>.csv    # one file per sample (0-based index k)
├── METASIMULATIONS/
│   └── <data_type>/opendata/
│       ├── real/               # real-network simulation pickles
│       ├── <model>/            # reconstructed-network simulation pickles
│       ├── real_prevalence.pkl # mean per-basin prevalence (real net, written by step 3)
│       └── rec_prevalence_<model>.pkl  # per-basin prevalence (rec nets, written by step 4)
├── OUT_DATA/
│   └── metapopulation_epidemics_<data_type>_opendata_<model>.dat  # written by step 2
├── FIGURES/                    # output plots (created by step 5)
├── LOGS/                       # run logs (created by run_metapopulation_simulations.sh)
├── cartopy_data/               # local Natural Earth shapefiles (optional, avoids downloads)
│
├── metapopulation_simulations_on_realnet.py
├── metapopulation_simulations_on_recnet.py
├── statistics_on_realnet_simulations.py
├── statistics_on_recnet_simulations.py
├── plot_metapopulation_simulations.py
└── run_metapopulation_simulations.sh
```

---

## Pipeline overview

```
Step 1  metapopulation_simulations_on_realnet.py
          ↓ METASIMULATIONS/.../real/real_graph_SIR_*_seed<hub>_sim<NNN>.pkl

Step 2  metapopulation_simulations_on_recnet.py  [per model]
          ↓ METASIMULATIONS/.../<model>/rec_graph_SIR_*_net<NNNN>_seed<hub>[_sim<NNN>].pkl

Step 3  statistics_on_realnet_simulations.py
          ↓ prints pairwise metrics (baseline stochastic spread)
          ↓ METASIMULATIONS/.../real_prevalence.pkl

Step 4  statistics_on_recnet_simulations.py  [per model]
          ↓ prints rec-vs-real metrics
          ↓ METASIMULATIONS/.../rec_prevalence_<model>.pkl

Step 5  plot_metapopulation_simulations.py  [per model]
          ↓ FIGURES/metapopulation_simulation_vs_time_seed<hub>_<model>_[S|I|R].png
          ↓ FIGURES/epidemics_on_map_<model>_realnet_seed<hub>.png
          ↓ FIGURES/epidemics_on_map_<model>_recnet<N>_seed<hub>.png
```

Steps 3 and 4 must come **after** steps 1 and 2 respectively, because they write the prevalence caches that step 5 reads.

---

## Quick start

Run the full pipeline with the provided shell script:

```bash
./run_metapopulation_simulations.sh --hub London --time 100 model_A model_B
```

| Option | Default | Description |
|---|---|---|
| `--hub <city>` | `London(UK)` | City name for the epidemic seed (matched against `basin_label`) |
| `--time <t>` | `80` | Time step at which to snapshot the map |
| `--data-type <type>` | `basin-WAN` | Data type label used in directory and file names |
| `--n-sims-real <N>` | `10` | Stochastic realizations per hub on the real network |
| `--n-sims-rec <N>` | `1` | Stochastic realizations per hub on each reconstructed network |
| `--log-dir <dir>` | `LOGS` | Root directory for log files |

Each step's stdout/stderr is written to `LOGS/run_<timestamp>/<step>.log` and also shown live in the terminal. The script aborts on the first failure.

---

## Running steps individually

### 1 — Real-network simulations

```bash
python metapopulation_simulations_on_realnet.py [--data-type basin-WAN] [--n-sims 10] \
    [--beta-rate 0.3] [--mu-rate 0.1] [--t-max 300] [--I0 10]
```

Produces one pickle per `(hub, realization)` under `METASIMULATIONS/<data_type>/opendata/real/`.

### 2 — Reconstructed-network simulations

```bash
python metapopulation_simulations_on_recnet.py <model> [--data-type basin-WAN] \
    [--samples-dir Ensembles] [--rec-list 9 97 24 ...] [--n-sims 1]
```

Reads edgelists from `Ensembles/<model>/edgelist_<k>.csv` (0-based index).  
Produces one pickle per `(network, hub, realization)` under `METASIMULATIONS/<data_type>/opendata/<model>/`.

### 3 — Statistical analysis on the real network

```bash
python statistics_on_realnet_simulations.py [--data-type basin-WAN]
```

Compares all pairs of stochastic realizations for each hub (baseline noise floor).  
Also writes `METASIMULATIONS/<data_type>/opendata/real_prevalence.pkl`.

### 4 — Statistical analysis on reconstructed networks

```bash
python statistics_on_recnet_simulations.py <model> [--data-type basin-WAN]
```

Compares every reconstructed simulation against every real-network realization.  
Also writes `METASIMULATIONS/<data_type>/opendata/rec_prevalence_<model>.pkl`.

### 5 — Plotting

```bash
python plot_metapopulation_simulations.py --model <model> [--hub "London(UK)"] \
    [--data_type basin-WAN] [--time 80]
```

`--hub` accepts a city name (case-insensitive match on `basin_label` in `basin_info.csv`); defaults to `London(UK)`.  
Requires both prevalence caches from steps 3 and 4 to exist.

---

## Metrics

Both statistics scripts report the following metrics averaged over all hub seeds:

| Metric | Description |
|---|---|
| Area between curves | ∫\|I_a(t) − I_b(t)\| dt on global prevalence |
| L2 distance | ‖I_a − I_b‖₂ on global prevalence |
| Peak time difference | \|argmax I_a − argmax I_b\| |
| Peak magnitude difference | \|max I_a − max I_b\| |
| Epidemic length difference | Difference in number of time steps above threshold |
| RMSE prevalence | Mean over time of per-basin RMSE |
| L2 entropy | ‖H_a − H_b‖₁ on spatial entropy of prevalence |

---

## Dependencies

```
numpy
pandas
scipy
matplotlib
seaborn
cartopy
networkx
pycountry_convert
epidemik
```

Install with:
```bash
pip install numpy pandas scipy matplotlib seaborn cartopy networkx pycountry-convert epidemik
```

The `cartopy_data/` folder can be pre-populated with Natural Earth shapefiles to avoid network downloads at render time.

---

## Known issues and patches

### `epidemik` 0.1.3.3 — `add_spontaneous` silently drops the rate

**Symptom:** `ValueError: pvals < 0, pvals > 1 or pvals contains NaNs` raised inside
`numpy.random.Generator.multinomial` during the first simulation timestep.

**Cause:** A typo in `epidemik/EpiModel.py` line 131. The recovery rate passed to
`add_spontaneous` is never stored because the line uses a subscript read instead of an
assignment:

```python
# buggy (epidemik 0.1.3.3)
self.params[rate_key, rate]   # no-op read — value is never written

# correct
self.params[rate_key] = rate
```

Every spontaneous transition (e.g. I → R) ends up with an unset rate, which resolves to
`None` at simulation time, becomes `NaN` in the probability array, and crashes the
multinomial sampler.

**Fix:** Edit the installed library file directly:

```
<path-to-env>/lib/python3.x/site-packages/epidemik/EpiModel.py
```

Find line 131 inside `add_spontaneous` and change:

```python
self.params[rate_key, rate]
```

to:

```python
self.params[rate_key] = rate
```
