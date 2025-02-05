#!/bin/bash

# Create a Python timing script
cat > timing_wrapper.py << 'EOL'
import timeit
import sys
import subprocess

def run_command():
    subprocess.run(sys.argv[1:], check=True)

number = 1
time_taken = timeit.timeit(run_command, number=number)
print(f"{time_taken:.3f}")  # Just print the raw time
EOL

# Initialize timing variables
total_time_regular=0
count_regular=0
total_time_gwot=0
count_gwot=0

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
            time=$(python timing_wrapper.py python main.py --N "$N" --srand "$srand" --lamda "$lamda" --outfile "$outfile")
            total_time_regular=$(echo "$total_time_regular + $time" | bc)
            count_regular=$((count_regular + 1))
            echo "Time: ${time}s"
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
            time=$(python timing_wrapper.py python main.py --gwot --N "$N" --srand "$srand" --lamda_gwot "$lamda" --outfile "$outfile")
            total_time_gwot=$(echo "$total_time_gwot + $time" | bc)
            count_gwot=$((count_gwot + 1))
            echo "Time: ${time}s"
        done
    done
done

# Calculate and display averages
echo -e "\n=== Timing Summary ==="
echo "Regular Method:"
echo "Total time: ${total_time_regular}s"
avg_time_regular=$(echo "scale=3; $total_time_regular / $count_regular" | bc)
echo "Average time per run: ${avg_time_regular}s"
echo "Number of runs: $count_regular"

echo -e "\nGWOT Method:"
echo "Total time: ${total_time_gwot}s"
avg_time_gwot=$(echo "scale=3; $total_time_gwot / $count_gwot" | bc)
echo "Average time per run: ${avg_time_gwot}s"
echo "Number of runs: $count_gwot"

echo -e "\nSpeed Comparison:"
speedup=$(echo "scale=3; $avg_time_regular / $avg_time_gwot" | bc)
echo "GWOT is ${speedup}x faster than Regular method"