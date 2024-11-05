# ridge_regression_simulation.py

from simulation_module import BaseSimulation
from sklearn.linear_model import Ridge
import numpy as np  # Added import statement

class RidgeRegressionSimulation(BaseSimulation):
    def initialize_dataset(self):
        """
        Initializes a dataset with correlated features.
        """
        X1 = np.random.rand(self.initial_points, 1) * 10
        X2 = X1 + np.random.randn(self.initial_points, 1)  # Correlated feature
        self.X = np.hstack((X1, X2))
        self.y = self.generate_target(self.X)

    def generate_target(self, X):
        """
        Generates a target with correlated features.
        """
        return 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(X.shape[0]) * 2

    def initialize_models(self):
        """
        Initializes a Ridge Regressor.
        """
        self.models = {
            'ridge': Ridge(alpha=1.0)
        }

def run_ridge_regression_simulation():
    sim = RidgeRegressionSimulation(
        simulation_id=8,
        np_seed=8,
        initial_points=19,
        new_points=21,
        window_size=15,
        anomaly_threshold_multiplier=2.2
    )
    sim.run()

if __name__ == "__main__":
    run_ridge_regression_simulation()
