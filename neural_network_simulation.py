# neural_network_simulation.py

from simulation_module import BaseSimulation
from sklearn.neural_network import MLPRegressor
import numpy as np  # Added import statement

class NeuralNetworkSimulation(BaseSimulation):
    def initialize_dataset(self):
        """
        Initializes a dataset with multiple features.
        """
        self.X = np.random.rand(self.initial_points, 2) * 10  # Two features
        self.y = self.generate_target(self.X)

    def generate_target(self, X):
        """
        Generates a target with multiple features and noise.
        """
        return 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(X.shape[0]) * 2

    def initialize_models(self):
        """
        Initializes a Neural Network Regressor.
        """
        self.models = {
            'neural_network': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=self.np_seed)
        }

    def update_plots(self, axs, plot_idx, ax_main, ax_residual, ax_mse, ax_window, ax_feature_importance, new_X, predictions, anomaly_color, iteration):
        """
        Overrides the base class method to handle multiple features in plotting.

        Parameters:
        - axs (array): Array of Axes objects.
        - plot_idx (int): Current plot index.
        - ax_main (Axes): Main plot for data and predictions.
        - ax_residual (Axes): Residual plot.
        - ax_mse (Axes): MSE over iterations plot.
        - ax_window (Axes): Rolling window data points plot.
        - ax_feature_importance (Axes): Feature importance plot (if applicable).
        - new_X (np.ndarray): New feature data point.
        - predictions (dict): Predictions from all models.
        - anomaly_color (str): Color indicating if the new point is an anomaly.
        - iteration (int): Current iteration number.
        """
        # Since it's 2D, use a scatter plot with color indicating predictions
        ax_main.clear()
        scatter = ax_main.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='viridis', label='Training Data')
        for model_name, y_pred in predictions.items():
            # Plotting predictions against the first two features for visualization
            ax_main.scatter(self.X[:, 0], self.X[:, 1], c=y_pred, cmap='coolwarm', alpha=0.5, label=f'{model_name.capitalize()} Prediction')
        # Highlight the new prediction
        for model_name, pred in predictions.items():
            if pred[-1] is not None:
                ax_main.scatter(new_X[0, 0], new_X[0, 1], c=anomaly_color, marker='x', s=100, label='New Data Prediction')
        ax_main.set_xlabel('Feature 1 (X1)')
        ax_main.set_ylabel('Feature 2 (X2)')
        ax_main.set_title(f'Simulation {self.simulation_id}: Neural Network Regression')
        ax_main.legend(loc='upper left')

        # Residual Plot for Neural Network
        residuals = self.y - predictions['neural_network']
        ax_residual.clear()
        ax_residual.stem(range(len(residuals)), residuals, linefmt='r-', markerfmt='ro', basefmt='b-')
        ax_residual.set_xlabel('Data Point Index')
        ax_residual.set_ylabel('Residuals')
        ax_residual.set_title('Residual Plot for Neural Network Regression')
        ax_residual.axhline(0, color='black', linewidth=0.5)

        # MSE Over Time
        ax_mse.clear()
        ax_mse.plot(
            self.metrics['iteration'],
            self.metrics['mse']['neural_network'],
            marker='o',
            color='purple',
            label='Neural Network MSE'
        )
        ax_mse.set_xlabel('Iteration')
        ax_mse.set_ylabel('Mean Squared Error')
        ax_mse.set_title('MSE Over Time')
        ax_mse.legend()

        # Rolling Window Data Points
        if len(self.X) > self.window_size:
            X_train = self.X[-self.window_size:]
            y_train = self.y[-self.window_size:]
        else:
            X_train = self.X
            y_train = self.y

        ax_window.clear()
        ax_window.scatter(X_train[:, 0], X_train[:, 1], color='blue', label='Rolling Window Data')
        ax_window.set_xlabel('Feature 1 (X1)')
        ax_window.set_ylabel('Feature 2 (X2)')
        ax_window.set_title('Data Points Used in Rolling Window')
        ax_window.legend()

        # Neural Network does not have feature importance in scikit-learn
        if ax_feature_importance:
            ax_feature_importance.clear()
            ax_feature_importance.text(0.5, 0.5, 'Feature Importance Not Available', horizontalalignment='center', verticalalignment='center')
            ax_feature_importance.set_title('Feature Importance')
            ax_feature_importance.axis('off')

        plt.tight_layout()
        plt.draw()
