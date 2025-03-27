from simulation_module import BaseSimulation
from sklearn.neural_network import MLPRegressor
import numpy as np


class NeuralNetworkSimulation(BaseSimulation):

    def initialize_dataset(self):
        """
        Initializes a dataset for neural network regression.
        Loads CSV if available; otherwise, generates synthetic 2D feature data.
        """
        if self.csv_path and self.feature_columns and self.target_column:
            super().initialize_dataset()
        else:
            self.X = np.random.rand(self.initial_points, 2) * 10  # Two features
            self.y = self.generate_target(self.X)

    def generate_target(self, X):
        """
        Generates a target using a multi-feature linear relationship with noise.
        """
        return 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(X.shape[0]) * 2

    def initialize_models(self):
        """
        Initializes a Neural Network Regressor.
        """
        self.models = {
            'neural_network': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=self.np_seed)
        }


def run_neural_network_simulation():
    sim = NeuralNetworkSimulation(
        simulation_id=4,
        np_seed=4,
        initial_points=15,
        new_points=20,
        window_size=15,
        anomaly_threshold_multiplier=2.0
    )
    sim.run()


if __name__ == "__main__":
    run_neural_network_simulation()
