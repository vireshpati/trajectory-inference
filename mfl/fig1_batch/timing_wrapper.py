import timeit
import sys
import subprocess

def run_command():
    subprocess.run(sys.argv[1:], check=True)

number = 1
time_taken = timeit.timeit(run_command, number=number)
print(f"{time_taken:.3f}")  # Just print the raw time
