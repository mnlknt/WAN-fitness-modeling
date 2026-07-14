# Block-Fitness Modeling of the Global Air Mobility Network

Code and data accompanying the paper:

> **Block-Fitness Modeling of the Global Air Mobility Network**
> Giulia Fischetti, Anna Mancini, Giulio Cimini, Jessica T. Davis, Abby Leung, Alessandro Vespignani, Guido Caldarelli
> arXiv:2601.13867 — https://arxiv.org/abs/2601.13867

This repository provides two connected pipelines:

1. **Network reconstruction** — fitness-based models that reconstruct the World Air Transportation Network (WAN) as an undirected weighted graph, using a generative maximum-entropy framework combining OpenSky-derived connectivity data with Wikipedia airport passenger flows. Ensembles of reconstructed networks are written to `Ensembles/<model>/`.
2. **Epidemic simulation and analysis** — runs SIR metapopulation epidemics on both the real network and the reconstructed networks, then compares them to quantify how well each reconstruction model preserves epidemic dynamics.

The reconstruction step (pipeline 1) must be run first to populate `Ensembles/` before the epidemic pipeline (pipeline 2) can be executed.

> **Note on the real-network simulations.** The script `metapopulation_simulations_on_realnet.py` requires the *real* (ground-truth) WAN as input. That network is derived from a proprietary dataset that we cannot redistribute, so it is **not** included in this repository. This script is provided as a reference implementation for users who already have access to their own real-network data in a compatible format; running the paper's exact real-network simulations from a clean clone of this repository is not possible. Pipelines 1 and 2 steps 2–5 (reconstruction and analysis on the reconstructed networks) are fully runnable from the data provided here.

## Models

The repository implements five reconstruction variants, selected via the `model` variable in `reconstruction_undirected.py`:

| Name      | α     | β | Description                                             |
|-----------|-------|---|---------------------------------------------------------|
| `model_K` | 1     | 0 | Configuration-like baseline (degree-driven)             |
| `model_S` | 0.80  | 0 | Sub-linear fitness model                                |
| `model_D` | 1     | 1 | Fitness + distance (gravity-like)                       |
| `model_B` | 1     | 0 | Block-fitness model (regional structure, no distance)   |
| `model_C` | 1.56  | 1 | Block-fitness model with distance (full model)          |

`model_B` and `model_C` use the regional block structure (`bdgm_functions.py`); the others use the basic fitness reconstruction (`rec_functions.py`).

## Repository structure

```
.
├── reconstruction_undirected.py   # Pipeline 1 entry point — network reconstruction
├── strength_flows_computation.py  # Region-level strengths and flows
├── rec_functions.py               # Standard fitness reconstruction (K, S, D)
├── bdgm_functions.py              # Block-fitness reconstruction (B, C)
│
├── metapopulation_simulations_on_realnet.py   # Pipeline 2, step 1
├── metapopulation_simulations_on_recnet.py    # Pipeline 2, step 2
├── statistics_on_realnet_simulations.py       # Pipeline 2, step 3
├── statistics_on_recnet_simulations.py        # Pipeline 2, step 4
├── plot_metapopulation_simulations.py         # Pipeline 2, step 5
├── run_metapopulation_simulations.sh          # Pipeline 2 orchestrator
│
├── Input/                         # Input data (CSV files, see below)
│   ├── opensky_el_final.csv
│   └── opensky_nodes_final.csv
├── Output/                        # Created automatically on first run (pipeline 1)
├── Ensembles/                     # Reconstructed network samples
│   └── <model>/
│       └── edgelist_<k>.csv       # One file per sample (0-based index)
├── METASIMULATIONS/               # Simulation pickles (pipeline 2)
├── FIGURES/                       # Output plots (pipeline 2)
├── LOGS/                          # Run logs (pipeline 2)
├── cartopy_data/                  # Local Natural Earth shapefiles (optional)
│
├── requirements.txt
├── LICENSE                        # MIT (covers the code only)
├── DATA_NOTICE.md                 # Attribution and license for the input data
└── README.md
```

## Installation

Tested with **Python 3.8.17**.

```bash
# 1. Clone the repo
git clone https://github.com/mnlknt/WAN-fitness-modeling.git
cd WAN-fitness-modeling

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

The `cartopy_data/` folder can be pre-populated with Natural Earth shapefiles to avoid network downloads during map rendering in pipeline 2.

## Pipeline 1 — Network reconstruction

Open `reconstruction_undirected.py` and edit the configuration block near the top:

```python
model   = 'model_C'   # one of: 'model_K', 'model_S', 'model_D', 'model_B', 'model_C'
density = 'basins'    # 'basins' (real density) or 'opensky' (OpenSky density)
N       = 10          # number of network realizations in the ensemble
```

Then run:

```bash
python reconstruction_undirected.py
```

Paths are resolved relative to the script's own location, so it works regardless of the directory you launch it from.

### Reconstruction output

Files are written to `Output/` (created automatically). For a given `<model>`:

- `strength_by_region_opensky.csv` — node strength split by destination region
- `ws_regions.csv` — total inter-region strength matrix
- `ls_regions.csv` — inter-region link counts
- `haversine_distances_basins.csv` — pairwise great-circle distances between basins
- `z_<model>.csv` — fitted Lagrange multiplier(s) for the chosen model
- `rec_variables_<model>.csv` — per-pair link probabilities and expected weights
- `Ensembles/<model>/edgelist_<i>.csv` — `N` sampled network realizations

Intermediate outputs are reused across runs: if `z_<model>.csv` or `rec_variables_<model>.csv` already exist, they're loaded from disk instead of recomputed. Delete them to force a fresh run.

Repeat this pipeline for each model you want to evaluate; `Ensembles/` will accumulate one subfolder per model.

## Pipeline 2 — Epidemic simulation and analysis

Run the full simulation pipeline with the provided shell script:

```bash
./run_metapopulation_simulations.sh --hub London --time 100 model_B model_C
```

| Option | Default | Description |
|---|---|---|
| `--hub <city>` | `London(UK)` | City name for the epidemic seed (matched against `basin_label`) |
| `--time <t>` | `80` | Time step at which to snapshot the map |
| `--data-type <type>` | `basin-WAN` | Data type label used in directory and file names |
| `--n-sims-real <N>` | `10` | Stochastic realizations per hub on the real network |
| `--n-sims-rec <N>` | `1` | Stochastic realizations per hub on each reconstructed network |
| `--log-dir <dir>` | `LOGS` | Root directory for log files |

Each step's stdout/stderr is written to `LOGS/run_<timestamp>/<step>.log` and also shown live. The script aborts on the first failure.

### Running pipeline 2 steps individually

**Step 1 — Real-network simulations**

> ⚠️ **Requires proprietary real-network data not included in this repository.** This step can only be executed by users who already hold a copy of the ground-truth WAN in a compatible format. See the note at the top of this README.
>
> If you have your own real-network data and want to plug it in, place the following two files in an `IN_DATA/` folder at the repository root:
>
> - **`IN_DATA/basin_el_final.csv`** — undirected edge list of the real network. Required columns:
>   - `basin_id1`, `basin_id2` — integer basin identifiers for the two endpoints
>   - `weight` — passenger flow between them (float)
>
>   The script symmetrizes the edges internally, so provide each edge once.
>
> - **`IN_DATA/basin_info.csv`** — per-basin metadata. Required columns:
>   - `basin_id` — integer basin identifier (must match the ids used in the edge list)
>   - `basin_label` — human-readable city/basin name (used for output labels and hub matching)
>   - `country_iso3` — ISO 3166-1 alpha-3 country code (used to derive the continent)
>   - `basin_population` — basin population (integer or float)
>
> With those two files in place, the script below runs end-to-end and produces one pickle per `(hub, realization)` under `METASIMULATIONS/<data_type>/opendata/real/`.

```bash
python metapopulation_simulations_on_realnet.py [--data-type basin-WAN] [--n-sims 10] \
    [--beta-rate 0.3] [--mu-rate 0.1] [--t-max 300] [--I0 10]
```

Produces one pickle per `(hub, realization)` under `METASIMULATIONS/<data_type>/opendata/real/`.

**Step 2 — Reconstructed-network simulations**

```bash
python metapopulation_simulations_on_recnet.py <model> [--data-type basin-WAN] \
    [--samples-dir Ensembles] [--rec-list 9 97 24 ...] [--n-sims 1]
```

Reads edgelists from `Ensembles/<model>/edgelist_<k>.csv` (0-based index).
Produces one pickle per `(network, hub, realization)` under `METASIMULATIONS/<data_type>/opendata/<model>/`.

**Step 3 — Statistical analysis on the real network**

```bash
python statistics_on_realnet_simulations.py [--data-type basin-WAN]
```

Compares all pairs of stochastic realizations for each hub (baseline noise floor).
Writes `METASIMULATIONS/<data_type>/opendata/real_prevalence.pkl`.

**Step 4 — Statistical analysis on reconstructed networks**

```bash
python statistics_on_recnet_simulations.py <model> [--data-type basin-WAN]
```

Compares every reconstructed simulation against every real-network realization.
Writes `METASIMULATIONS/<data_type>/opendata/rec_prevalence_<model>.pkl`.

**Step 5 — Plotting**

```bash
python plot_metapopulation_simulations.py --model <model> [--hub "London(UK)"] \
    [--data_type basin-WAN] [--time 80]
```

`--hub` accepts a case-insensitive match on `basin_label` in the basin info file. Requires both prevalence caches from steps 3 and 4 to exist.

### Metrics reported by pipeline 2

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

## Input data

The `Input/` folder must contain two CSVs:

- **`opensky_el_final.csv`** — Edge list of observed flights between basins. Columns: `basin1`, `basin2`. Aggregated and anonymized from the [OpenSky Network COVID-19 flight-list dataset](https://doi.org/10.5281/zenodo.7923702) on Zenodo.
- **`opensky_nodes_final.csv`** — Node attributes. Columns: `basin_id`, `regions`, `latitude`, `longitude`, `pax_strength`. Basin definitions and their geographic coordinates come from the [**GLEAMviz**](http://www.gleamviz.org/) global metapopulation simulator; the `pax_strength` column is aggregated from Wikipedia airport passenger data.

The fit coefficients α and β used inside `strength_flows_computation.py` are hard-coded from the values reported in the paper. See `DATA_NOTICE.md` for the full attribution and licensing terms for all data sources.

## Citation

If you use this code or data, please cite the paper:

```bibtex
@misc{fischetti2026blockfitness,
  title         = {Block-Fitness Modeling of the Global Air Mobility Network},
  author        = {Fischetti, Giulia and Mancini, Anna and Cimini, Giulio and
                   Davis, Jessica T. and Leung, Abby and Vespignani, Alessandro and
                   Caldarelli, Guido},
  year          = {2026},
  eprint        = {2601.13867},
  archivePrefix = {arXiv},
  primaryClass  = {physics.soc-ph},
  url           = {https://arxiv.org/abs/2601.13867}
}
```

Please also cite the **OpenSky Network** (flight connectivity) and **GLEAMviz** (basin geography) — see `DATA_NOTICE.md` for exact references.

## Known issues and patches

### `epidemik` 0.1.3.3 — `add_spontaneous` silently drops the rate

**Symptom:** `ValueError: pvals < 0, pvals > 1 or pvals contains NaNs` raised inside `numpy.random.Generator.multinomial` during the first simulation timestep.

**Cause:** A typo in `epidemik/EpiModel.py` line 131. The recovery rate passed to `add_spontaneous` is never stored because the line uses a subscript read instead of an assignment:

```python
# buggy (epidemik 0.1.3.3)
self.params[rate_key, rate]   # no-op read — value is never written

# correct
self.params[rate_key] = rate
```

Every spontaneous transition (e.g. I → R) ends up with an unset rate, which resolves to `None` at simulation time, becomes `NaN` in the probability array, and crashes the multinomial sampler.

**Fix:** Edit the installed library file directly at `<path-to-env>/lib/python3.x/site-packages/epidemik/EpiModel.py`, find line 131 inside `add_spontaneous`, and change `self.params[rate_key, rate]` to `self.params[rate_key] = rate`.

## License

- **Code:** MIT License — see [`LICENSE`](LICENSE).
- **Data:** The input CSVs are derived from third-party sources with their own terms — see [`DATA_NOTICE.md`](DATA_NOTICE.md).
