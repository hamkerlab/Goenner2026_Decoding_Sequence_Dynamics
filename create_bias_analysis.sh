#!/usr/bin/env bash
# Run the per-cell-rate sim N_RUNS times per n_cells.
# Requires rate_seed=None in sim_params_seq.py so the seed is drawn fresh each time.
# Otherwise, the random seed will only step forward for poisson draws and reinitialize for a new run.

MODE="${1:-smooth}"
N_RUNS="${2:-30}"
N_CELLS=(100 200 400 600 800 1000 1400 2000)

SIM="sim_sequence_percell_rates.py"
ANALYSIS="analysis_step_size_bias.py"

for run in $(seq 1 "$N_RUNS"); do
  for nc in "${N_CELLS[@]}"; do
    echo "=== run ${run}/${N_RUNS} | n_cells=${nc} | mode=${MODE} ==="
    python "$SIM" "$nc" "$MODE" --run "$run"
  done
done

echo "=== all sims done -> aggregating bias over runs ==="
python "$ANALYSIS" "${N_CELLS[@]}"

echo "done."