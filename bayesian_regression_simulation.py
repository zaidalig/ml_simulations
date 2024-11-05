# bayesian_regression_simulation.py

from simulation_module import BaseSimulation
from sklearn.linear_model import BayesianRidge
import numpy as np  # Added import statement

class BayesianRegressionSimulation(BaseSimulation):
    def initialize_dataset(self):
        """
        Initializes a dataset with outliers.
        """
        self.X = np.random.rand(self.initial_points, 1) * 10
        self.y = self.generate_target(self.X)
        # Introduce outliers
        num_outliers = int(0.1 * self.initial_points)
        if num_outliers > 0:
            indices = np.random.choice(range(self.initial_points), num_outliers, replace=False)
            self.y[indices] += np.random.randn(num_outliers) * 15  # Large noise for outliers

    def generate_target(self, X):
        """
        Generates a linear relationship with noise.
        """
        return 2.5 * X.squeeze() + np.random.randn(X.shape[0]) * 2

    def initialize_models(self):
        """
        Initializes a Bayesian Ridge Regressor.
        """
        self.models = {
            'bayesian': BayesianRidge()
        }

def run_bayesian_regression_simulation():
    sim = BayesianRegressionSimulation(
        simulation_id=7,
        np_seed=7,
        initial_points=16,
        new_points=24,
        window_size=12,
        anomaly_threshold_multiplier=2.0
    )
    sim.run()

if __name__ == "__main__":
    run_bayesian_regression_simulation()
