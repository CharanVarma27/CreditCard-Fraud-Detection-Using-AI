# AI-Powered Credit Card Fraud Detection System  
**Author:** Pachamatla Charan Sai Venkata Varma  
**University:** Anurag University – Dept. of Electronics & Computer Engineering  
**Project Date:** 2025  

## Project Overview  
This project presents a full-stack machine learning system to detect fraudulent credit-card transactions.  
It uses a hybrid approach combining supervised learning (XGBoost), unsupervised anomaly detection (Isolation Forest, LOF), imbalance handling (SMOTE), and explainable AI (SHAP).  
A Streamlit dashboard and Flask API make the solution deployment-ready.

## Why This Matters  
- With the growth of online payments, fraud attempts are rising rapidly.  
- Traditional rule-based systems are less effective against evolving fraud patterns.  
- This system flags suspicious transactions in real time and provides human-readable explanations, enabling analysts to act quickly.

## Tools & Technologies  
- **Programming:** Python  
- **ML Libraries:** Scikit-Learn, XGBoost, Imbalanced-Learn  
- **Explainability:** SHAP  
- **Frontend:** Streamlit  
- **Backend:** Flask REST API  
- **Visualization:** Matplotlib, Seaborn  
- **Dataset:** Kaggle – “Credit Card Fraud Detection (Europe, 2013)”  
- **Deployment:** Model saved via joblib, API endpoint, dashboard UI  

## Project Structure  
```
<Insert the folder structure you used>
```

## Steps Involved  
1. **Data Preprocessing** – Scale `Time` and `Amount`, drop time/amount after scaling.  
2. **Exploratory Data Analysis (EDA)** – check missing values, class imbalance, feature distributions.  
3. **Feature Understanding** – The dataset uses PCA-transformed variables `V1`…`V28` to preserve confidentiality while capturing transaction behavior. Example: `V14`, `V17`, `V12` were found to be high-impact in fraud detection.  
4. **Handling Class Imbalance** – Only ~0.17% of transactions are fraud. SMOTE used to balance training data.  
5. **Modelling**  
   - *Isolation Forest* and *LOF* for anomaly detection.  
   - *XGBoost Classifier* as the high-performance model.  
   - Key parameters: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`.  
6. **Evaluation** – Use precision, recall, F1-score, ROC-AUC. Emphasis on high recall (reducing missed frauds).  
7. **Explainability** – Apply SHAP to understand feature contributions at both global (top features) and local (single transaction) levels.  
8. **Deployment**  
   - Model saved (`.pkl` file).  
   - Flask API: `/predict` endpoint expecting JSON payload, returns fraud probability and label.  
   - Streamlit UI: Accepts manual input or CSV upload, visualizes results and SHAP plots, handles smooth UI transitions.  

## Sample Results  
- XGBoost achieved **ROC-AUC ≈ 0.99**, **Recall > 95%** in testing.  
- Top features influencing fraud detection: `V14`, `V17`, `V12`, `Scaled_Amount`.  
- Example: A sudden large foreign purchase triggers high `V14` + low density in anomaly detectors → flagged as fraud.  
*(Insert confusion-matrix screenshot here)*  
*(Insert SHAP summary plot here)*  

## How to Run Locally  
1. Clone the repository:  
   ```bash
   git clone https://github.com/YourUsername/CreditCard-FraudDetection.git
   cd CreditCard-FraudDetection
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Download the dataset (link to Kaggle) and place it in `data/`.  
4. Run the Streamlit dashboard:  
   ```bash
   cd app
   streamlit run dashboard.py
   ```  
5. (Optional) Run the Flask API in a separate terminal:  
   ```bash
   python api.py
   ```  

## Folder Contents  
- `data/` – transaction CSV (only include sample or link due to size)  
- `notebooks/` – EDA & model training notebook  
- `models/` – trained model files (`.pkl`)  
- `app/` – dashboard and API code  
- `reports/` – final report PDF  
- `requirements.txt`, `.gitignore`, `LICENSE`  

## License  
This project is released under the [MIT License](LICENSE).

## 📬 Contact  
For any questions or collaboration, reach me at: **charansai2707@gmail.com**  
Feel free to connect on LinkedIn: www.linkedin.com/in/charan-sai-venkata-varma

