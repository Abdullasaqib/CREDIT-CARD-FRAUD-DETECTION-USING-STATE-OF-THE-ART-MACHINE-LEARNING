# CryptoGuard: Entity-Centric Fraud Detection

**CryptoGuard** is a robust, machine-learning-powered platform designed to detect fraudulent transactions in Cryptocurrency and Ethereum datasets. It features an entity-centric approach, identifying not just individual suspicious transactions but also the high-risk accounts ("entities") behind them.

## 🚀 Key Features

*   **Dynamic Data Upload**: Drag-and-drop any CSV dataset (e.g., PaySim, Ethereum Transaction Data).
*   **Auto-ML Pipeline**: Automatically detects the target column, cleans data, handles class imbalance, and trains an **XGBoost** model.
*   **Entity-Centric Analysis**: Aggregates fraud indicators to identify specific high-risk Accounts or Users.
*   **Smart Training**:
    *   **Anti-Leakage**: Automatically drops ID columns (like `nameOrig`) from training to prevent overfitting/memorization.
    *   **Optimization**: Uses smart subsampling for fast hyperparameter tuning on large datasets.
*   **Interactive Dashboard**:
    *   **Fraud Distribution Pie Chart**: Visualizes the "Total Stolen Amount" by Entity.
    *   **Feature Importance**: Explains *why* the model flagged a transaction.
    *   **Confusion Matrix**: Shows model reliability (True Positives vs False Alarms).

## 🛠️ Tech Stack

*   **Backend**: FastAPI, Python, Pandas, XGBoost, Scikit-Learn
*   **Frontend**: HTML5, CSS3 (Cyber/Dark Theme), Vanilla JavaScript, Chart.js
*   **Deployment**: Uvicorn Server

## ⚙️ Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-repo/crypto-fraud-detection.git
    cd crypto-fraud-detection
    ```

2.  **Create Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Running the Application

1.  **Start the Server**
    ```bash
    uvicorn app.main:app --reload
    ```

2.  **Open Dashboard**
    *   Go to `http://127.0.0.1:8000` in your browser.
    *   Upload your CSV file (e.g., `transaction_dataset.csv`).
    *   Wait for the analysis to complete (Loader will indicate progress).
    *   Explore the dashboard results!

## 📊 How It Works

1.  **Upload**: User uploads a CSV.
2.  **Auto-Detect**: The system scans the file to find the Target Column (e.g., `isFraud`, `Class`) and separates "Meta Columns" (IDs, Names) from "Feature Columns".
3.  **Train**: An XGBoost classifier is trained on the Features. ID columns are excluded to ensure the model learns *patterns*, not *specific users*.
4.  **detect**: The trained model predicts fraud probabilities on the dataset.
5.  **Aggregate**: High-risk transactions are grouped by their Entity ID to calculate the total fraudulent amount per person.
6.  **Visualize**: Results are sent to the frontend for interactive visualization.

## 📁 Project Structure

```
crypto_fraud_detection/
├── app/
│   ├── main.py              # App Entry Point
│   ├── api.py               # API Routes (/analyze)
│   ├── services/
│   │   ├── cleaning.py      # Data Cleaning Logic
│   │   └── modeling.py      # XGBoost & Evaluation Logic
│   └── models.py
├── static/
│   ├── index.html           # Dashboard UI
│   ├── style.css            # Styling
│   └── app.js               # Frontend Logic
├── uploads/                 # Temporary file storage
└── requirements.txt
```
