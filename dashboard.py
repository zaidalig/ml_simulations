import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px

# Import simulation classes
from linear_regression_simulation import LinearRegressionSimulation
from decision_tree_simulation import DecisionTreeSimulation
from random_forest_simulation import RandomForestSimulation
from neural_network_simulation import NeuralNetworkSimulation
from gradient_boosting_simulation import GradientBoostingSimulation
from elastic_net_simulation import ElasticNetRegressionSimulation
from knn_regression_simulation import KNNRegressionSimulation
from bayesian_regression_simulation import BayesianRegressionSimulation
from lasso_regression_simulation import LassoRegressionSimulation
from ridge_regression_simulation import RidgeRegressionSimulation
from polynomial_regression_simulation import PolynomialRegressionSimulation

st.set_page_config(layout="wide")
st.title("📊 Machine Learning Simulation Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Preview Uploaded Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["number"]).columns

    with st.form("column_selection"):
        st.markdown("### 🧠 Select Simulation and Columns")

        simulation_type = st.selectbox("Select Simulation", [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Neural Network",
            "Gradient Boosting",
            "Elastic Net",
            "KNN Regression",
            "Bayesian Regression",
            "Lasso Regression",
            "Ridge Regression",
            "Polynomial Regression"
        ])

        target_column = st.selectbox("Select Target Column (y)", numeric_cols)
        feature_candidates = [col for col in numeric_cols if col != target_column]
        feature_columns = st.multiselect("Select Feature Columns (X)", feature_candidates)

        submit = st.form_submit_button("Run Simulation")

    # Optional visualization preferences
    with st.expander("📌 Optional: Choose Visualizations"):
        selected_viz = st.multiselect(
            "Select what you'd like to visualize after simulation:",
            ["Scatter Plot", "Residual Plot", "Feature vs Target", "Model Comparison"],
            default=["Scatter Plot", "Residual Plot", "Model Comparison"]
        )

    if submit and feature_columns and target_column:
        csv_path = "uploaded_dataset.csv"
        df.to_csv(csv_path, index=False)

        st.success("✅ CSV Uploaded Successfully. Starting Simulation...")

        simulation_class = {
            "Linear Regression": LinearRegressionSimulation,
            "Decision Tree": DecisionTreeSimulation,
            "Random Forest": RandomForestSimulation,
            "Neural Network": NeuralNetworkSimulation,
            "Gradient Boosting": GradientBoostingSimulation,
            "Elastic Net": ElasticNetRegressionSimulation,
            "KNN Regression": KNNRegressionSimulation,
            "Bayesian Regression": BayesianRegressionSimulation,
            "Lasso Regression": LassoRegressionSimulation,
            "Ridge Regression": RidgeRegressionSimulation,
            "Polynomial Regression": PolynomialRegressionSimulation
        }.get(simulation_type)

        sim = simulation_class(
            simulation_id=99,
            np_seed=42,
            initial_points=len(df),
            new_points=0,
            window_size=min(20, len(df)),
            anomaly_threshold_multiplier=2.0,
            csv_path=csv_path,
            feature_columns=feature_columns,
            target_column=target_column
        )
        sim.run()

        st.markdown("### 📈 Simulation Metrics")
        try:
            with open("simulation_99_metrics.json", "r") as f:
                metrics = json.load(f)

            st.subheader("Mean Squared Error (MSE)")
            st.line_chart(pd.DataFrame(metrics['mse']))

            st.subheader("R² Score")
            st.line_chart(pd.DataFrame(metrics['r2']))

            st.subheader("Anomaly Flags")
            st.dataframe(pd.DataFrame({
                "Iteration": metrics["iteration"],
                "Anomaly": metrics["anomalies"]
            }))
        except Exception as e:
            st.warning(f"Metrics not found or failed to load: {e}")

        # ✅ Model Comparison Bar Chart
        if "Model Comparison" in selected_viz:
            mse_scores = {model: scores[-1] for model, scores in metrics["mse"].items()}
            r2_scores = {model: scores[-1] for model, scores in metrics["r2"].items()}

            mse_df = pd.DataFrame(list(mse_scores.items()), columns=["Model", "MSE"])
            r2_df = pd.DataFrame(list(r2_scores.items()), columns=["Model", "R2"])

            st.subheader("📊 Model Comparison (MSE)")
            fig_mse = px.bar(mse_df, x="Model", y="MSE", color="Model", text="MSE", height=400)
            st.plotly_chart(fig_mse, use_container_width=True)

            st.subheader("📊 Model Comparison (R² Score)")
            fig_r2 = px.bar(r2_df, x="Model", y="R2", color="Model", text="R2", height=400)
            st.plotly_chart(fig_r2, use_container_width=True)

        # ✅ Scatter Plot
        if "Scatter Plot" in selected_viz:
            st.subheader("📉 Scatter Plot: True vs Predicted")
            for model_name in sim.models:
                y_true = sim.y
                y_pred = sim.predict_models(sim.X)[model_name]
                scatter_df = pd.DataFrame({
                    "Actual": y_true,
                    "Predicted": y_pred
                })
                fig = px.scatter(scatter_df, x="Actual", y="Predicted", title=f"{model_name.capitalize()} Predictions vs True Values")
                st.plotly_chart(fig, use_container_width=True)

        # ✅ Residual Plot
        if "Residual Plot" in selected_viz:
            st.subheader("📉 Residual Plot")
            for model_name in sim.models:
                y_true = sim.y
                y_pred = sim.predict_models(sim.X)[model_name]
                residuals = y_true - y_pred
                residual_df = pd.DataFrame({
                    "Index": range(len(residuals)),
                    "Residual": residuals
                })
                fig = px.scatter(residual_df, x="Index", y="Residual", title=f"Residuals of {model_name.capitalize()}")
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

        # ✅ Optional Feature vs Target plot
        if "Feature vs Target" in selected_viz and len(feature_columns) == 1:
            st.subheader("📊 Feature vs Target")
            x = sim.X[:, 0]
            y = sim.y
            ft_df = pd.DataFrame({feature_columns[0]: x, target_column: y})
            fig = px.scatter(ft_df, x=feature_columns[0], y=target_column, title=f"{feature_columns[0]} vs {target_column}")
            st.plotly_chart(fig, use_container_width=True)

        # ✅ Download predictions
        export_path = f"simulation_99_predictions.csv"
        if os.path.exists(export_path):
            with open(export_path, "rb") as f:
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=f.read(),
                    file_name=os.path.basename(export_path),
                    mime="text/csv"
                )

        # ✅ Logs
        st.markdown("### 📝 Simulation Logs")
        if os.path.exists("simulation_99.log"):
            with open("simulation_99.log", "r") as log_file:
                st.text_area("Logs", log_file.read(), height=300)
        else:
            st.info("No log file found.")
else:
    st.warning("📂 Please upload a CSV file to begin.")
