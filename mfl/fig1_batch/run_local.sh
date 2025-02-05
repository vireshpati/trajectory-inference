#!/bin/bash

# Create a Python timing script
cat > timing_wrapper.py << 'EOL'
import timeit
import sys
import subprocess

def run_command():
    subprocess.run(sys.argv[1:], check=True)

# Number of repetitions for timing (adjust as needed)
number = 1
time_taken = timeit.timeit(run_command, number=number)
print(f"Average time: {time_taken/number:.3f} seconds")
EOL

# Regular method
lamda_vals=(0.025 0.1)
N_vals=(64 16 4)
srand_vals=($(seq 1 3))

echo "Starting Regular Method runs..."

for lamda in "${lamda_vals[@]}"; do
    for N in "${N_vals[@]}"; do
        for srand in "${srand_vals[@]}"; do 
            outfile="out_N_${N}_srand_${srand}_lamda_${lamda}.npy"
            echo "Running: N=$N, srand=$srand, lamda=$lamda"
            python timing_wrapper.py python main.py --N "$N" --srand "$srand" --lamda "$lamda" --outfile "$outfile"
        done
    done
done

# GWOT method
lamda_gwot_vals=(0.00125 0.005)

echo -e "\nStarting GWOT Method runs..."

for lamda in "${lamda_gwot_vals[@]}"; do
    for N in "${N_vals[@]}"; do
        for srand in "${srand_vals[@]}"; do 
            outfile="out_gwot_N_${N}_srand_${srand}_lamda_${lamda}.npy"
            echo "Running: N=$N, srand=$srand, lamda=$lamda"
            python timing_wrapper.py python main.py --gwot --N "$N" --srand "$srand" --lamda_gwot "$lamda" --outfile "$outfile"
        done
    done
done