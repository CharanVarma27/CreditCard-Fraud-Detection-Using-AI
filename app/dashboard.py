# ===================================================
# STAGE 9 - ENHANCED AI FRAUD DETECTION DASHBOARD (UPGRADED)
# ===================================================
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import json
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="💳 AI Fraud Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# --- CUSTOM CSS FOR A MODERN UI ---
st.markdown("""
<style>
    /* General Styles */
    body { color: #E0E0E0; }
    .stApp { background: linear-gradient(180deg, #0f2027, #203a43, #2c5364); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #2c5364; }
    .metric-card { background-color: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.3); border: 1px solid rgba(255, 255, 255, 0.1); }
</style>
""", unsafe_allow_html=True)

# --- SHAP PLOT FUNCTION ---
def st_shap(plot, height=None):
    """A function to display SHAP plots in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- MAIN DASHBOARD LAYOUT ---
st.title("💳 AI-Powered Credit Card Fraud Detection")
st.markdown("### Analyze transactions in real-time and understand model decisions with Explainable AI.")

# --- CREATE TABS FOR DIFFERENT FUNCTIONALITIES ---
tab1, tab2, tab3 = st.tabs(["**🔬 Single Transaction Analysis**", "**🗂️ Batch File Analysis**", "**📊 Dataset Insights**"])

# =====================================================================================
# TAB 1: SINGLE TRANSACTION ANALYSIS
# =====================================================================================
with tab1:
    st.header("Analyze a Single Transaction")
    st.markdown("Enter transaction details manually to get an instant fraud prediction and explanation.")

    with st.form("transaction_form"):
        st.sidebar.header("🧾 Transaction Input")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            V4 = st.number_input("Feature V4", value=4.7, help="A PCA-transformed feature from the transaction.")
            V10 = st.number_input("Feature V10", value=-2.5, help="A PCA-transformed feature from the transaction.")
        with col2:
            V12 = st.number_input("Feature V12", value=-6.2, help="A PCA-transformed feature from the transaction.")
            V14 = st.number_input("Feature V14", value=-8.5, help="A PCA-transformed feature from the transaction.")
        with col3:
            V17 = st.number_input("Feature V17", value=-4.8, help="A PCA-transformed feature from the transaction.")
            # --- IMPROVEMENT 1: User-friendly amount input ---
            Amount = st.number_input("Transaction Amount ($)", value=150.00, help="The actual transaction amount in dollars.")
        
        submitted = st.form_submit_button("🚀 Analyze Transaction", use_container_width=True)

    if submitted:
        # --- BEHIND THE SCENES SCALING ---
        # We scale the amount here using the mean and std dev from the original dataset
        # This is more user-friendly than asking for a "scaled" value
        mean_amount = 88.34 
        std_amount = 250.12
        Scaled_Amount = (Amount - mean_amount) / std_amount
        
        feature_defaults = {'V1':-0.01, 'V2':0.06, 'V3':0.17, 'V5':-0.05, 'V6':-0.27, 'V7':0.04, 'V8':0.02, 'V9':-0.05, 'V11':-0.03, 'V13':-0.01, 'V15':0.04, 'V16':0.0, 'V18':-0.0, 'V19':0.0, 'V20':-0.06, 'V21':-0.02, 'V22':0.0, 'V23':-0.01, 'V24':0.04, 'V25':0.01, 'V26':0.0, 'V27':0.0, 'V28':0.0, 'Scaled_Time':0.0}
        
        user_input = {
            'V4': V4, 'V10': V10, 'V12': V12, 'V14': V14, 'V17': V17, 'Scaled_Amount': Scaled_Amount,
            **feature_defaults
        }

        with st.spinner("🧠 Analyzing with AI... Please wait."):
            try:
                # [The rest of your Tab 1 code remains the same]
                response = requests.post("http://127.0.0.1:5000/predict", json=user_input, timeout=10)
                response.raise_for_status()
                result = response.json()
                prob = result['fraud_probability']
                is_fraud = result['is_fraud']
                shap_data = result['shap_values']

                st.subheader("💡 Prediction Results")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    if is_fraud: st.error("### ⚠️ High Risk: Fraudulent Transaction Detected!")
                    else: st.success("### ✅ Low Risk: Transaction Appears Legitimate")
                    st.markdown("</div>", unsafe_allow_html=True)
                with res_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=prob, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Fraud Probability Score"}, gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "darkred" if is_fraud else "green"}, 'steps': [{'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.3)'}, {'range': [50, 75], 'color': 'rgba(255, 255, 0, 0.3)'}, {'range': [75, 100], 'color': 'rgba(255, 0, 0, 0.3)'}]}))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.subheader("🤖 Explainable AI (XAI) Insights")
                st.markdown("The chart below shows which features pushed the model's prediction towards 'Fraud' (red arrows) or 'Legitimate' (blue arrows).")
                shap_values_obj = shap.Explanation(values=np.array(shap_data['values']), base_values=shap_data['base_value'], data=np.array(shap_data['features']), feature_names=shap_data['feature_names'])
                st_shap(shap.force_plot(shap_values_obj, matplotlib=False), 400)
            except requests.exceptions.RequestException as e:
                st.error(f"API connection failed! Ensure the Flask API is running. Error: {e}")

# =====================================================================================
# TAB 2: BATCH FILE ANALYSIS
# =====================================================================================
with tab2:
    # [Your existing Tab 2 code remains the same]
    st.header("Analyze a Batch of Transactions")
    st.markdown("Upload a CSV file with multiple transactions to get predictions for each one.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("First 5 rows of your uploaded data:")
        st.dataframe(data.head())
        if st.button("📊 Analyze Batch File", use_container_width=True):
            with st.spinner('Preprocessing data and analyzing transactions...'):
                processed_data = data.copy()
                if 'Amount' in processed_data.columns and 'Time' in processed_data.columns:
                    scaler = StandardScaler()
                    processed_data['Scaled_Amount'] = scaler.fit_transform(processed_data[['Amount']])
                    processed_data['Scaled_Time'] = scaler.fit_transform(processed_data[['Time']])
                    processed_data.drop(['Time', 'Amount'], axis=1, inplace=True)
                    if 'Class' in processed_data.columns:
                        processed_data.drop(['Class'], axis=1, inplace=True)
                else:
                    st.error("Uploaded CSV must contain 'Time' and 'Amount' columns.")
                    st.stop()
                predictions = []
                for i, row in processed_data.iterrows():
                    payload = row.to_dict()
                    try:
                        response = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=10)
                        if response.status_code == 200:
                            predictions.append(response.json())
                        else:
                            predictions.append({'is_fraud': 'Error', 'fraud_probability': 'N/A'})
                    except requests.exceptions.RequestException:
                        predictions.append({'is_fraud': 'API Error', 'fraud_probability': 'N/A'})
                results_df = pd.DataFrame(predictions)
            st.success("✅ Batch analysis complete!")
            fraud_count = (results_df['is_fraud'] == True).sum()
            st.metric("Total Fraudulent Transactions Detected", f"{fraud_count} out of {len(data)}")
            display_df = pd.concat([data.reset_index(drop=True), results_df[['is_fraud', 'fraud_probability']]], axis=1)
            st.dataframe(display_df.style.apply(lambda row: ['background-color: #993333'] * len(row) if row.is_fraud == True else [''] * len(row), axis=1))
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            csv = convert_df_to_csv(display_df)
            st.download_button(label="📥 Download Results as CSV", data=csv, file_name='fraud_detection_results.csv', mime='text/csv', use_container_width=True)

# =====================================================================================
# TAB 3: DATASET INSIGHTS (UPGRADED)
# =====================================================================================
with tab3:
    st.header("Insights from the Training Dataset")
    st.markdown("Visualizations and statistics from the original Kaggle Credit Card Fraud dataset.")
    
    try:
        df = pd.read_csv('../data/creditcard.csv')
        
        # --- ADD-ON 2: Explanation of 'V' Features ---
        st.subheader("What are the 'V' Features?")
        st.info("""
        The `V1` to `V28` features are the result of a mathematical transformation called **Principal Component Analysis (PCA)**. 
        
        **Why was this done?** The original dataset contained sensitive information which could not be shared. PCA was used to transform this sensitive data into a set of numerical components while preserving the underlying patterns. 
        
        Think of it as creating a detailed summary of a book without using any of the original words. Each 'V' feature represents a key pattern or 'theme' from the original data, which is essential for the AI to detect fraud.
        """)
        
        # --- ADD-ON 3: Interactive Feature Explorer ---
        st.subheader("How 'V' Features Distinguish Fraud")
        st.markdown("Select a feature below to see how its values are distributed for fraudulent vs. legitimate transactions. If the two distributions look very different, the feature is a strong predictor of fraud.")
        
        feature_list = [f'V{i}' for i in range(1, 29)]
        selected_feature = st.selectbox("Select a feature to explore:", feature_list, index=13) # Default to V14

        # Create the plot
        fraud_data = df[df['Class'] == 1][selected_feature]
        legit_data = df[df['Class'] == 0][selected_feature]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=legit_data, name='Legitimate', marker_color='#4CAF50', opacity=0.7))
        fig.add_trace(go.Histogram(x=fraud_data, name='Fraudulent', marker_color='#D32F2F', opacity=0.7))
        
        fig.update_layout(
            barmode='overlay',
            title_text=f'Distribution of {selected_feature} for Fraud vs. Legitimate Transactions',
            xaxis_title_text=f'Value of {selected_feature}',
            yaxis_title_text='Count',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            font={'color': "white"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Observation:** For feature **{selected_feature}**, you can see how the values for fraudulent transactions (in red) are distributed differently from the legitimate ones (in green). This difference is what the AI model learns to use for its predictions.")
        
    except FileNotFoundError:
        st.error("`data/creditcard.csv` not found. Please place the dataset in the correct directory.")

# --- FOOTER ---
st.markdown("---")
st.markdown("<center>Built by <i><b>Charan Varma</b></i>", unsafe_allow_html=True)