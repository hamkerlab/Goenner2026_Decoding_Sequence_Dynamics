n_cells=( 200)

for n in "${n_cells[@]}"
do
    echo "Running with n_cells = $n"
    python3 sim_sequence_percell_rates.py $n "smooth" > /dev/null &
    python3 sim_sequence_percell_rates.py $n "speedmod" > /dev/null
    wait
    python3 comp1_addpositions_hilbert.py $n > /dev/null
    python3 comp3_ricedist_speedmod_const.py $n > /dev/null
    python3 analysis_spearman_phase.py $n --ref-line > /dev/null 
done
