# decision_tree_simulation.py

from simulation_module import BaseSimulation
from sklearn.tree import DecisionTreeRegressor
import numpy as np  # Added import statement

class DecisionTreeSimulation(BaseSimulation):
    def initialize_dataset(self):
        """
        Initializes a quadratic dataset.
        """
        self.X = np.random.rand(self.initial_points, 1) * 10
        self.y = self.generate_target(self.X)

    def generate_target(self, X):
        """
        Generates a quadratic relationship with noise.
        """
        y = 1.5 * X.squeeze()**2 + np.random.randn(X.shape[0]) * 5
        if isinstance(y, np.ndarray) and y.size == 1:
            return float(y)
        return y

    def initialize_models(self):
        """
        Initializes a Decision Tree Regressor.
        """
        self.models = {
            'tree': DecisionTreeRegressor(max_depth=4)
        }

def run_decision_tree_simulation():
    sim = DecisionTreeSimulation(
        simulation_id=2,
        np_seed=2,
        initial_points=12,
        new_points=25,
        window_size=20,
        anomaly_threshold_multiplier=2.5
    )
    sim.run()

if __name__ == "__main__":
    run_decision_tree_simulation()
