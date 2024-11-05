# knn_regression_simulation.py

from simulation_module import BaseSimulation
from sklearn.neighbors import KNeighborsRegressor
import numpy as np  # Added import statement

class KNNRegressionSimulation(BaseSimulation):
    def initialize_dataset(self):
        """
        Initializes a logarithmic dataset.
        """
        self.X = np.linspace(1, 10, self.initial_points).reshape(-1, 1)
        self.y = self.generate_target(self.X)

    def generate_target(self, X):
        """
        Generates a logarithmic relationship with noise.
        """
        y = 3 * np.log(X.squeeze()) + np.random.randn(X.shape[0]) * 1.5
        if isinstance(y, np.ndarray) and y.size == 1:
            return float(y)
        return y


    def initialize_models(self):
        """
        Initializes a K-Nearest Neighbors Regressor.
        """
        self.models = {
            'knn': KNeighborsRegressor(n_neighbors=5)
        }

def run_knn_regression_simulation():
    sim = KNNRegressionSimulation(
        simulation_id=5,
        np_seed=5,
        initial_points=20,
        new_points=25,
        window_size=15,
        anomaly_threshold_multiplier=2.1
    )
    sim.run()

if __name__ == "__main__":
    run_knn_regression_simulation()
