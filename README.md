# Goenner2026_Decoding_Sequence_Dynamics

Simulates hippocampal place cell activity during spatial sequences and analyzes Bayesian decoding accuracy.

The code in this repository is used to investigate whether place-cell sequences exhibit step-like movement (discrete attractor dynamics) or continuous movement (continuous attractor dynamics). We show that the previous decoding approach is prone to artifacts and propose a modified method to better distinguish continuous from discrete dynamics in place-cell sequences.

## Data Generation

`sim_sequence.py` - Baseline simulation with fixed 2.5 cm steps, 1000 repetitions.

`sim_sequence_vartotspeed.py` - Variable step sizes from lognormal distribution, 100 repetitions.

`sim_sequence_paramfile.py` - Simulation with parameters from external file, overlapping windows, random trajectory offsets.


`speedmod_sim_sequence.py` - Speed modulation with fixed steps.

`speedmod_vartotspeed_sequence.py` - Speed modulation with variable baseline speeds.

`speedmod_sim_sequence_paramfile.py` - Speed modulation with overlapping windows and offsets.


`azizi_newInitLearn.py` - Spiking place cell network initialization (only used for imports).

`ann_contextosc_nodelay.py` - Goal-directed navigation network with oscillatory dynamics.

## Data Analysis

`ann_bayes_decode.py` - Bayesian decoding of ANNarchy simulation spike trains.

`comp1_newlayout_speedmod_const.py` - Compares speed-modulated vs. constant speed decoding accuracy.

`comp1_addpositions_hilbert.py` - Phase-dependent analysis using Hilbert transform.

`comp3_ricedist_speedmod_const.py` - Fits Rice and Rayleigh distributions to decoded step sizes.

`step_size_comparison_simplified.R` - Statistical analysis comparing step sizes across conditions using regression and circular statistics.

## Utilities

`sim_params_seq.py` - Parameter configuration file defining simulation settings for different parameter sets.

`plotting_func.py` - Plotting functions for trajectories, firing maps, and decoding results.

`circ_stats.py` - Circular statistics functions (circular-linear correlation).

`new_colormaps.py` - Colormap definitions

`pytest.py` - Helper function for loading numpy pickle files (in R).

`py_hilbert.py` - Wrapper for scipy Hilbert transform (for use with R reticulate).

## Dependencies
  
The neurosimulator ANNarchy 4.8 was used for data generation. Please note that ANNarchy is only supported on the following operating systems: 

- GNU/Linux
- MacOS X
- Windows (inside WSL2)

For detailed information, please visit the [ANNarchy repository](https://github.com/ANNarchy/ANNarchy).

Additional dependencies are:

- **Python**: NumPy, SciPy, Matplotlib, scikit-learn, brian, Python 2.7
- **R**: Rcpp, RcppCNPy, reticulate, circular, ggplot2, car, lmtest, estimatr.
