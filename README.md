# Block-Fitness Modeling of the Global Air Mobility Network

Code accompanying the paper:

> **Block-Fitness Modeling of the Global Air Mobility Network**
> Giulia Fischetti, Anna Mancini, Giulio Cimini, Jessica T. Davis, Abby Leung, Alessandro Vespignani, Guido Caldarelli
> arXiv:2601.13867 — https://arxiv.org/abs/2601.13867

This repository reconstructs the **World Air Transportation Network (WAN)** as an undirected weighted graph, using a generative maximum-entropy framework that combines OpenSky-derived connectivity data with Wikipedia airport passenger flows. Several null and fitness-based models are implemented for comparison.

## Models

The repository implements five variants, selected via the `model` variable in `reconstruction_undirected.py`:

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
├── reconstruction_undirected.py   # Main entry point — run this
├── strength_flows_computation.py  # Region-level strengths and flows
├── rec_functions.py               # Standard fitness reconstruction (K, S, D)
├── bdgm_functions.py              # Block-fitness reconstruction (B, C)
├── Input/                         # Input data (CSV files, see below)
│   ├── opensky_el_final.csv
│   └── opensky_nodes_final.csv
├── Output/                        # Created automatically on first run
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

## Usage

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

## Output

Files are written to `Output/` (created automatically). For a given `<model>`:

- `strength_by_region_opensky.csv` — node strength split by destination region
- `ws_regions.csv` — total inter-region strength matrix
- `ls_regions.csv` — inter-region link counts
- `haversine_distances_basins.csv` — pairwise great-circle distances between basins
- `z_<model>.csv` — fitted Lagrange multiplier(s) for the chosen model
- `rec_variables_<model>.csv` — per-pair link probabilities and expected weights
- `Ensembles/<model>/edgelist_<i>.csv` — `N` sampled network realizations

Intermediate outputs are reused across runs: if `z_<model>.csv` or `rec_variables_<model>.csv` already exist, they're loaded from disk instead of recomputed. Delete them to force a fresh run.

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

## License

- **Code:** MIT License — see [`LICENSE`](LICENSE).
- **Data:** The input CSVs are derived from third-party sources with their own terms — see [`DATA_NOTICE.md`](DATA_NOTICE.md).
