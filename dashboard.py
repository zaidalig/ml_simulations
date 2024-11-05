# dashboard.py

import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(sim_id):
    filepath = f'simulation_{sim_id}_metrics.json'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def load_logs(sim_id):
    log_file = f'simulation_{sim_id}.log'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return f.read()
    return "No logs available."

def main():
    st.title("Real-Time Simulation Dashboard")

    simulation_id = st.sidebar.selectbox("Select Simulation ID", list(range(1, 11)))

    metrics = load_metrics(simulation_id)
    if metrics:
        st.header(f"Simulation {simulation_id} Metrics")

        # MSE Plot
        mse_df = pd.DataFrame(metrics['mse'])
        st.subheader("Mean Squared Error Over Time")
        st.line_chart(mse_df)

        # R2 Plot
        r2_df = pd.DataFrame(metrics['r2'])
        st.subheader("R-squared Score Over Time")
        st.line_chart(r2_df)

        # Anomalies
        anomalies = pd.DataFrame({
            'Iteration': metrics['iteration'],
            'Anomaly': metrics['anomalies']
        })
        st.subheader("Anomaly Detection")
        st.dataframe(anomalies[anomalies['Anomaly']])

        # Display Logs
        st.subheader("Simulation Logs")
        logs = load_logs(simulation_id)
        st.text_area("Logs", logs, height=300)
    else:
        st.warning("No metrics found for the selected simulation.")

if __name__ == "__main__":
    main()
