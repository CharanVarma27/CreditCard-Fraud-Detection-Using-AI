# ===================================================
# FINAL FLASK API FOR FRAUD DETECTION MODEL
# ===================================================
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import shap  # <-- Import SHAP

app = Flask(__name__)

try:
    # Load the trained model
    model = joblib.load('../models/xgboost_fraud_model.pkl')
    
    # Initialize the SHAP explainer with the loaded model
    explainer = shap.TreeExplainer(model)
    
    # Define the expected order of columns based on the training data
    model_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                     'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                     'Scaled_Amount', 'Scaled_Time']

    print("✅ Model and SHAP explainer loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model or explainer: {e}")
    model = None
    explainer = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    try:
        json_data = request.get_json()
        
        # Create a DataFrame and reorder columns to match the model's training order
        df_from_json = pd.DataFrame([json_data])
        input_df = df_from_json[model_columns]
        
        # --- MAKE PREDICTION ---
        fraud_probability = model.predict_proba(input_df)[0][1]
        is_fraud = bool(fraud_probability > 0.5)
        
        # --- GENERATE SHAP EXPLANATION (This was the missing part) ---
        shap_values = explainer.shap_values(input_df)
        
        shap_explanation = {
            "base_value": float(explainer.expected_value),
            "values": shap_values[0].tolist(),
            "features": input_df.iloc[0].tolist(),
            "feature_names": model_columns
        }
        
        # --- CONSTRUCT THE FULL RESPONSE ---
        response = {
            "is_fraud": is_fraud,
            "fraud_probability": round(fraud_probability * 100, 2),
            "shap_values": shap_explanation  # <-- Add SHAP values to the response
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"❌ An error occurred during prediction: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)