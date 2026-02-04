# Achtung:
# Wird die Staerke des Artefakts u.U. vom Verhaeltnis der
# echten Bewegungsgeschwindigkeit zur Gesamtzahl der Spikes
# bestimmt?!
# D.h. der Artefakt waere maximal fuer geringe Geschwindigkeiten
# und geringe Spike-Counts?!
# Vergleich von id 2 und id 6 zeigt:
# ----------------------------------------------------------------------------------------------------------------------
# Geringe Spike-Counts (reduziert um Faktor 4 fuer id 6) erzeugen die Probleme, die von der Theorie her erwartet werden!
# ----------------------------------------------------------------------------------------------------------------------
#
# Wichtiger Faktor: Modulationstiefe!!!

# Partialkorrelation hilft fuer ALLE Parameter!
# ---------------------------------------------------------------------
#
# Uebrigens: Die Theorie fuer "smooth movement" ist nur korrekt fuer
# Overlap = 0 ! 
# Andernfalls sind aufeinanderfolgende Ortsschaetzungen nicht unabhaengig,
# und die Voraussetzungen fuer die Rice-Verteilung sind nicht erfuellt

# Anderer Ansatz:
# Starte die Simulation mit einem zufaelligen Phasenwert fuer jede Wiederholung!

params_id = 4 # 14 # 13 # 12 # 11 # 10 # 9 # 8 # 7 # 6 # 5 # 4 # 3 # 2 # 1

# Achtung: Abweichende Feuerraten zwischen "smooth" und "speed-modulated"?!
# Siehe comp_diagnostics.py !

# Gute Ergebnisse: 4, 5, evtl. 6
# Kriterien: 
# In comp3_ricedist, deutliche Korrelation zwischen Spike count und Step size (vector length),
# aber schwache Korrelation zwischen Spike count und x-step!
#
# Interessant bei 9: Asymmetrische Verteilung der x-steps um den wahren Wert!!! (Grosser Window-Overlap!)
# --------------------------------------------------------------------------

if params_id == 14: 
    # Like id 10, but with 10% overlap
	# & non-integer no. of n_osc_cycles
	# Result: 
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.01                                                                      # Decoding window length
	n_reps 	  			= 1000 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	#max_rate_Hz         = 0.5 * rate_factor * 300.0 # 300.0
	max_rate_Hz         = 2*0.5 * rate_factor * 300.0 # 300.0
    # CAUTION: Adjusted max_rate_Hz x 1.5 to compensate lower mean rates for "smooth" condition!!!

	n_mincells          = 1 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.1                                                                  # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 4.75 # 5	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction


if params_id == 13: 
    # Like id 10, but with 25% overlap
	# & non-integer no. of n_osc_cycles
	# Result: Naja...
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.01                                                                      # Decoding window length
	n_reps 	  			= 1000 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = 0.5 * rate_factor * 300.0 # 300.0
	#max_rate_Hz         = 1.5*0.5 * rate_factor * 300.0 # 300.0
    # CAUTION: Adjusted max_rate_Hz x 1.5 to compensate lower mean rates for "smooth" condition!!!

	n_mincells          = 1 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.25                                                                  # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 4.75 # 5	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction


elif params_id == 12:
	# Testing with zero overlap, DeltaT=0.005
	# & non-integer no. of n_osc_cycles
	# 
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.005                                                                      # Decoding window length
	n_reps 	  			= 1000 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = 0.5 * rate_factor * 300.0 # 300.0
	#max_rate_Hz         = 2 * 0.5 * rate_factor * 300.0 # 300.0
    # CAUTION: Adjusted max_rate_Hz x 2 to compensate lower mean rates for "smooth" condition!!!

	n_mincells          = 10 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.0                                                                  # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 4.75 # 5	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------


elif params_id == 11:
    # Like id 10, but increasing n_mincells to improve unbiased estimates!!!
	# Testing with zero overlap, DeltaT=0.01 (also reported by Pfeiffer & Foster)
	# & non-integer no. of n_osc_cycles
	# 
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.01                                                                      # Decoding window length
	n_reps 	  			= 1000 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = 0.5 * rate_factor * 300.0 # 300.0
	#max_rate_Hz         = 2 * 0.5 * rate_factor * 300.0 # 300.0
    # CAUTION: Adjusted max_rate_Hz x 2 to compensate lower mean rates for "smooth" condition!!!

	n_mincells          = 10 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.0                                                                  # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 4.75 # 5	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------

elif params_id == 10:
	# Testing with zero overlap, DeltaT=0.1 (also reported by Pfeiffer & Foster)
	# & non-integer no. of n_osc_cycles
	# 
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.01                                                                      # Decoding window length
	n_reps 	  			= 1000 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = 0.25 * rate_factor * 300.0 # 300.0
	#max_rate_Hz         = 4 * 0.25 * rate_factor * 300.0 # 300.0
    # CAUTION: Adjusted max_rate_Hz x 4 to compensate lower mean rates for "smooth" condition!!!

	n_mincells          = 1 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.0                                                                  # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 4.75 # 5	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------


elif params_id == 9:
	# Like id=8, but 5ms window shift - the default used by Pfeiffer & Foster (2015)
	# 
	# 
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.02                                                                      # Decoding window length
	n_reps 	  			= 1000 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = 0.1 * rate_factor * 300.0 # 300.0
	#max_rate_Hz         = 3 * 0.1 * rate_factor * 300.0 # 300.0
    # CAUTION: Adjusted max_rate_Hz x 3 to compensate lower mean rates for "smooth" condition!!!

	n_mincells          = 1 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.75                                                                  # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 4 # 5	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------
elif params_id == 8:
	# Like id=7, but reduced firing rates
	# 
	# 
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.02                                                                      # Decoding window length
	n_reps 	  			= 1000 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = 0.1 * rate_factor * 300.0 # 300.0
	n_mincells          = 1 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.5 # 0.9                                                                 # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 4 # 5	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------


elif params_id == 7:
	# Like id=2, but with DeltaT=20ms
	# 
	# 
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.02                                                                      # Decoding window length
	n_reps 	  			= 1000 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = 0.25 * rate_factor * 300.0 # 300.0
	n_mincells          = 1 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.5 # 0.9                                                                 # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 4 # 5	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------


elif params_id == 6:
	# Like id=2, but with DeltaT=10ms / 5ms ?! Firing rate adapted
	# 
	# 
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.01 # 0.02                                                               # Decoding window length
	n_reps 	  			= 1000 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = 0.25 * rate_factor * 300.0 # 300.0
	n_mincells          = 1 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.5 # 0.9                                                                 # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 4 # 5	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------


elif params_id == 5:
	# Like id=4, but with n_cycles = 5 - GOOD!
	# 
	# 
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.005 # 0.01 # 0.02                                                               # Decoding window length
	n_reps 	  			= 100 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = 0.25 * rate_factor * 300.0 # 0.5*rate_factor
	n_mincells          = 1 # 0 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.5 # 0.5 # 0.9                                                                 # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 5.0	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------

elif params_id == 4:
	# New: Used with random displacement of start locations [in the range +/- 5cm]
	# 
	# 
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.005 # 0.01 # 0.02                                                               # Decoding window length
	n_reps 	  			= 1000 	                                                            # no. of sequence repetitions (default: 1000)
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = 0.5 * rate_factor * 300.0 # 0.5*rate_factor
	#max_rate_Hz         = 1 * rate_factor * 300.0 # 0.5*rate_factor # compensation for lower firing rates in smooth movement
	n_mincells          = 1 # 0 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.5 # 0.5 # 0.9                                                                 # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 5.2	                                                                	# Number of simulated oscillation cycles
	# Caution: Do I get problems in the polar plots, which assume "equal" phase values across oscillation cycles?!
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------

elif params_id == 3:
	# 
	n_grid_cm 		  	= 200 	                                                       
	DeltaT_window_sec 	= 0.01 # 0.02                                                          
	n_reps 	  			= 100 # 100 	                                                       
	sigma_code_cm 		= 15.0                                                                 
	rate_factor 		= 25.0 / 3.125				
	max_rate_Hz         = rate_factor * 300.0 # 300.0
	n_mincells          = 1 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                  
	overlap			    = 0.75 # 0.5 # 0.9                                                                                                                                           
    # Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max	    = (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 	    	= int(n_windows_max) 	                                                    # decoding windows for a single simulated sequence
	n_osc_cycles 		= 4 # 5	                                                                    # Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------

elif params_id == 2:
	# New: Used with random displacement of start locations [in the range +/- 5cm]
	# Looking good, but:
	# Check the "L2 of mean steps" calculation?!
	# Update: Polar plots look much better when using the Hilbert transform to calculate phase values!
	n_grid_cm 		    = 200 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.01 # 0.02                                                               # Decoding window length
	n_reps 	  			= 1000 # 100 	                                                            # no. of sequence repetitions
	sigma_code_cm 		= 15.0                                                                      # Place field width and spatial spread of the code
	rate_factor 		= 25.0 / 3.125								    							# To maintain average spike counts constant - but they are about 2x higher now!
	max_rate_Hz         = rate_factor * 300.0 # 300.0
	n_mincells          = 1 # 10
    # Simulation parameters chosen a priori:
	x_speed_m_per_sec	= 10.0                                                                      # Sequence speed. Lower speeds allow to fit a higher number of oscillation cycles into the arena, 
                                                                                                    # and to fit more decoding windows into a fixed-duration oscillation cycle
	overlap			    = 0.5 # 0.9                                                                 # Overlap between successive decoding windows (equals 1 - window increment): 
                                                                                                    # Higher overlap allows to fit more decoding windows into a fixed-duration oscillation cycle
                                                                                                    # Zero overlap appears to increase the likelihood of "negative" steps    
	# Resulting simulation properties:
	T_max_sec           = 0.01 * n_grid_cm / x_speed_m_per_sec                                      # Max. simulation time determined by sequence progression speed and arena size
	n_windows_max		= (T_max_sec - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1   # Max. window count determined by simulation time, window size, and window overlap
	n_windows 		    = int(n_windows_max) 	                                            		# decoding windows for a single simulated sequence
	n_osc_cycles 		= 4 # 5	                                                                	# Number of simulated oscillation cycles
	frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
	x_step_cm           = 100 * x_speed_m_per_sec / frames_per_sec 	                                # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

# ---------------------------------------------------------------------

elif params_id == 1:
	# Params used for initial simulations:
	n_grid_cm 		    = 100 	                                                                    # Environment size: n_grid_cm x n_grid_cm
	DeltaT_window_sec 	= 0.02                                                                    	# Decoding window length
	n_reps 	  		    = 100 	                                                                # no. of sequence repetitions
	sigma_code_cm 		=  15.0                                                                     # Place field width and spatial spread of the code
	max_rate_Hz         = 300.0
	n_mincells          = 10

    # Simulation parameters chosen a priori:
	overlap			    =   0.0                                                                     # Overlap between successive decoding windows (equals 1 - window increment)
	n_windows 		    =  32 	                                                                    # decoding windows for a single simulated sequence
	n_osc_cycles 		=   2 	                                                                    # Number of simulated oscillation cycles
	x_step_cm	        =   2.5	                                                                    # Displacement per frame, in x-direction
	y_step_cm           = x_step_cm                                                                 # Displacement per frame, in y-direction

    # Resulting simulation properties:
	T_sim = DeltaT_window_sec + (n_windows - 1) * (1-overlap) * DeltaT_window_sec
	x_step_total_cm = n_windows * x_step_cm
	x_speed_m_per_sec = 0.01 * x_step_total_cm / T_sim



# ---------------------------------------------------------------------

print("")

frames_per_sec		= (1.0 - DeltaT_window_sec) / ((1-overlap) * DeltaT_window_sec) + 1
T_sim = DeltaT_window_sec + (n_windows - 1) * (1-overlap) * DeltaT_window_sec
T_osc = T_sim / n_osc_cycles

#print "Frames per second = ", frames_per_sec
print("Overlap = ", overlap)
print("Oscillation frequency = ", 1.0 / T_osc)
print("Frames per cycle = ", (T_osc - DeltaT_window_sec) / ((1 - overlap) * DeltaT_window_sec) + 1)
#print("FPC = ",   (1 + n_windows) / n_osc_cycles  - 1 # Special case for overlap = 0.5)
print("n_windows = ", n_windows)
print("Sequence speed [m/sec] = ", x_speed_m_per_sec)
print("x_step_cm = ", x_step_cm)
print("Simulated sequence duration [sec] = ", T_sim)
#print("Simulated sequence length [m] = ", T_sim * x_speed_m_per_sec)
print("Environment size [cm] = ", n_grid_cm)


