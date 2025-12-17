# CryptoGuard: Entity-Centric Fraud Detection
### A Machine Learning Approach to Cryptocurrency Security

---

# 1. The Problem

*   **Rising Crypto Fraud**: Cryptocurrency transactions are irreversible, making fraud detection critical.
*   **Complex Patterns**: Traditional rule-based systems fail to catch sophisticated laundering schemes.
*   **Entity Masking**: It's difficult to link individual fraudulent transactions back to a single malicious actor (Entity).

---

# 2. The Solution: CryptoGuard

**CryptoGuard** is an automated analytics platform that uses **XGBoost** to detect anomalies in transaction data.

### Key Innovations:
1.  **Dynamic Data Handling**: Works with user-uploaded CSVs with varying schemas.
2.  **Entity Awareness**: Identifies *Who* is committing fraud, not just *Which* transaction is bad.
3.  **Anti-Leakage Architecture**: Smartly separates ID columns from feature columns to ensure valid learning.

---

# 3. System Architecture

*   **Frontend**: Modern, Interactive Dashboard (HTML/CSS/JS).
*   **Backend API**: Fast & Async processing with **FastAPI**.
*   **ML Engine**: **XGBoost** with automated Hyperparameter Tuning.
*   **Data Processing**: **Pandas** for robust cleaning and encoding.

---

# 4. How It Works (The Pipeline)

1.  **Input**: User uploads a raw dataset (e.g., PaySim).
2.  **Auto-Discovery**: System scans for Target Variable (`isFraud`) and Meta Columns (`nameOrig`).
3.  **Training**: 
    *   Splits data into Training/Testing.
    *   Optimizes parameters on a subset (Speed).
    *   Trains full model on identified features.
4.  **Entity Mapping**: Maps high-risk predictions back to Account IDs.
5.  **Visualization**: Displays metrics, confusion matrix, and "Stolen Amount by Entity".

---

# 5. Key Features

*   **🔍 Feature Importance**: Explains the "Why" behind detections (e.g., "High amount moved in short time").
*   **📉 Fraud Distribution**: break-down of total fraud amount by specific Accounts.
*   **⚡ High Performance**: Uses subsampling for tuning, handling millions of rows in seconds.
*   **🛡️ Robustness**: Handles missing values and unknown categorical data automatically.

---

# 6. Conclusion

CryptoGuard bridges the gap between raw transaction data and actionable security insights. By focusing on **Entities** and using **Adaptive ML**, it provides a powerful tool for financial auditors and crypto exchanges.

### Thank You
**[Your Name/Team Name]**
