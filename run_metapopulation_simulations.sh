#!/usr/bin/env bash
# =============================================================================
# run_metapopulation_simulations.sh  —  Full WAN metapopulation simulations pipeline
#
# Usage:
#   ./run_metapopulation_simulations.sh [options] model1 [model2 ...]
#
# Options:
#   --data-type <type>   Data type label                          (default: basin-WAN)
#   --hub <city>         City name for the map seed               (required)
#   --time <t>           Map snapshot time step                   (default: 80)
#   --n-sims-real <N>    Stochastic realizations per hub, real    (default: 10)
#   --n-sims-rec  <N>    Stochastic realizations per hub, rec     (default: 1)
#   --log-dir <dir>      Root directory for logs                  (default: LOGS)
#
# Example:
#   ./run_metapopulation_simulations.sh --hub "London(UK)" --time 100 --n-sims-real 20 model_A model_B
# =============================================================================
set -euo pipefail

# ---------- usage ----------
usage() {
    echo "Usage: $0 [--data-type TYPE] --hub CITY [--time T] [--n-sims-real N] [--n-sims-rec N] [--log-dir DIR] model1 [model2 ...]"
    exit 1
}

# ---------- defaults ----------
DATA_TYPE="basin-WAN"
HUB="London(UK)"
MAP_TIME=80
N_SIMS_REAL=10
N_SIMS_REC=1
LOG_DIR="LOGS"
MODELS=()

# ---------- argument parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-type)   DATA_TYPE="$2";   shift 2 ;;
        --hub)         HUB="$2";         shift 2 ;;
        --time)        MAP_TIME="$2";    shift 2 ;;
        --n-sims-real) N_SIMS_REAL="$2"; shift 2 ;;
        --n-sims-rec)  N_SIMS_REC="$2";  shift 2 ;;
        --log-dir)     LOG_DIR="$2";     shift 2 ;;
        --help|-h)     usage ;;
        --*)           echo "Error: unknown option '$1'" >&2; usage ;;
        *)             MODELS+=("$1"); shift ;;
    esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: at least one model must be provided." >&2; usage
fi
if [[ -z "$HUB" ]]; then
    echo "Error: --hub <city> is required." >&2; usage
fi
if [[ ! -f "IN_DATA/basin_info.csv" ]]; then
    echo "Error: IN_DATA/basin_info.csv not found. Run this script from the project root." >&2
    exit 1
fi

# ---------- log directory ----------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${LOG_DIR}/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
PIPELINE_LOG="${RUN_DIR}/metapopulation_simulations.log"

echo "=============================================="
echo "Run directory : $RUN_DIR"
echo "Data type     : $DATA_TYPE"
echo "Models        : ${MODELS[*]}"
echo "Map hub       : $HUB  (t = $MAP_TIME)"
echo "Sims / hub    : real=$N_SIMS_REAL  rec=$N_SIMS_REC"
echo "=============================================="
echo ""

# ---------- step helper ----------
run_step() {
    local label="$1"; shift
    local log="${RUN_DIR}/${label}.log"
    echo "=== [$(date '+%H:%M:%S')] START: $label ===" | tee -a "$PIPELINE_LOG"
    if "$@" 2>&1 | tee "$log"; then
        echo "--- [$(date '+%H:%M:%S')] OK:    $label ---" | tee -a "$PIPELINE_LOG"
    else
        echo "!!! [$(date '+%H:%M:%S')] FAILED: $label — see $log" | tee -a "$PIPELINE_LOG"
        exit 1
    fi
    echo "" | tee -a "$PIPELINE_LOG"
}

# ---------- 1. Real-network simulations (model-independent) ----------
run_step "01_realnet_sim" \
    python metapopulation_simulations_on_realnet.py \
        --data-type "$DATA_TYPE" \
        --n-sims "$N_SIMS_REAL"

# ---------- 2. Reconstructed-network simulations (one run per model) ----------
for model in "${MODELS[@]}"; do
    run_step "02_recnet_sim_${model}" \
        python metapopulation_simulations_on_recnet.py "$model" \
            --data-type "$DATA_TYPE" \
            --n-sims "$N_SIMS_REC"
done

# ---------- 3. Statistical analysis on the real network ----------
run_step "03_realnet_stats" \
    python statistics_on_realnet_simulations.py \
        --data-type "$DATA_TYPE"

# ---------- 4. Statistical analysis on reconstructed networks ----------
for model in "${MODELS[@]}"; do
    run_step "04_recnet_stats_${model}" \
        python statistics_on_recnet_simulations.py "$model" \
            --data-type "$DATA_TYPE"
done

# ---------- 5. Figures ----------
for model in "${MODELS[@]}"; do
    run_step "05_plot_${model}" \
        python plot_metapopulation_simulations.py \
            --data_type "$DATA_TYPE" \
            --model "$model" \
            --hub "$HUB" \
            --time "$MAP_TIME"
done

echo ""
echo "Metapopulation simulations complete. All logs in ${RUN_DIR}/"
