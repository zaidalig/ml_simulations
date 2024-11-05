# svr_simulation.py

from simulation_module import BaseSimulation
from sklearn.svm import SVR
import numpy as np  # Added import statement

class SVRSimulation(BaseSimulation):
    def initialize_dataset(self):
        """
        Initializes an exponential dataset.
        """
        self.X = np.random.rand(self.initial_points, 1) * 10
        self.y = self.generate_target(self.X)

    def generate_target(self, X):
        """
        Generates an exponential relationship with noise.
        """
        return 2 * np.exp(0.3 * X.squeeze()) + np.random.randn(X.shape[0]) * 5

    def initialize_models(self):
        """
        Initializes a Support Vector Regressor.
        """
        self.models = {
            'svr': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        }

def run_svr_simulation():
    sim = SVRSimulation(
        simulation_id=4,
        np_seed=4,
        initial_points=18,
        new_points=22,
        window_size=20,
        anomaly_threshold_multiplier=2.2
    )
    sim.run()

if __name__ == "__main__":
    run_svr_simulation()
