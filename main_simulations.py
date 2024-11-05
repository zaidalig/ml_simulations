# main_simulations.py

import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_simulation_script(script_name):
    """
    Runs a simulation script using subprocess.

    Parameters:
    - script_name (str): The name of the simulation script file.
    """
    try:
        print(f"Starting {script_name}...")
        subprocess.run([sys.executable, script_name], check=True)
        print(f"Completed {script_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")

def main():
    # List of simulation script filenames with updated names
    simulation_scripts = [
        'linear_regression_simulation.py',
        'decision_tree_simulation.py',
        'random_forest_simulation.py',
        'svr_simulation.py',
        'knn_regression_simulation.py',
        'neural_network_simulation.py',
        'bayesian_regression_simulation.py',
        'ridge_regression_simulation.py',
        'lasso_regression_simulation.py',
        'gradient_boosting_simulation.py'
    ]

    # Verify all scripts exist
    existing_scripts = [s for s in simulation_scripts if os.path.exists(s)]
    missing_scripts = set(simulation_scripts) - set(existing_scripts)
    if missing_scripts:
        print(f"The following scripts are missing: {', '.join(missing_scripts)}\n")

    # Run simulations sequentially
    for script in existing_scripts:
        run_simulation_script(script)

    # Optionally, run simulations in parallel
    # Uncomment the following block to enable parallel execution
    """
    max_workers = 4  # Adjust based on your system's capabilities
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_simulation_script, script): script for script in existing_scripts}
        for future in as_completed(futures):
            script = futures[future]
            try:
                future.result()
                print(f"Completed {script}")
            except Exception as e:
                print(f"Error in {script}: {e}")
    """

if __name__ == "__main__":
    main()
