# CreditGuard: Advanced Crypto & Credit Card Fraud Detection

**CreditGuard** is a state-of-the-art, machine-learning-powered platform designed to detect fraudulent transactions in Cryptocurrency and Credit Card datasets. It leverages **XGBoost** with **Advanced Oversampling** and **Dynamic Thresholding** to ensure high sensitivity (Recall) and precision, identifying not just suspicious transactions but also the high-risk entities behind them.

---

## 🏗️ System Architecture

The system follows a modern client-server architecture with a fast asynchronous backend and a responsive frontend.

```mermaid
graph TD
    User[User] -->|Uploads CSV| Frontend[Frontend (HTML/JS)]
    Frontend -->|POST /analyze| API[FastAPI Backend]
    
    subgraph "Backend Services"
        API -->|1. Load & Detect Target| Cleaner[Data Cleaning Service]
        Cleaner -->|2. Preprocessing & Encoding| Modeler[Modeling Service]
        
        Modeler -->|3. Oversampling (SMOTE/Random)| TrainingSet[Balanced Training Data]
        TrainingSet -->|4. Train XGBoost| XGB[XGBoost Classifier]
        
        XGB -->|5. Predict Probabilities| Results[Result Generation]
        Results -->|6. Dynamic Thresholding| HighRisk[High Risk Candidates]
    end
    
    HighRisk -->|JSON Response| Frontend
    Frontend -->|Visualizes| Dashboard[Interactive Dashboard]
```

---

## 🚀 Key Features

*   **Universal Data Support**: Seamlessly handles diverse datasets (Credit Card `V1-V28`, Crypto `ETH/BTC`).
*   **Auto-ML Pipeline**:
    *   **Auto-Target Detection**: Automatically identifies the label column (e.g., `Class`, `isFraud`).
    *   **Smart Cleaning**: Handles missing values and encodings without data leakage.
*   **Advanced Fraud Detection Engine**:
    *   **Class Imbalance Handling**: Implements **Random Oversampling** to boost fraud signal in training.
    *   **XGBoost Classifier**: Uses gradient boosting for top-tier tabular performace.
    *   **Dynamic Thresholding**: Automatically lowers detection thresholds if standard strictness misses potential fraud, ensuring **Zero-Miss** capability.
*   **Comprehensive Reporting**:
    *   **Full Data Extraction**: Returns **ALL** detected fraud rows with every original column preserved.
    *   **Visual Analytics**: Interactive Pie Charts, Feature Importance graphs, and Confusion Matrices.

---

## 🔄 Workflow Data Architecture

The data flows through a rigorous pipeline to ensure accuracy and robustness.

```mermaid
sequenceDiagram
    participant User
    participant UI as Frontend
    participant API as Backend API
    participant ML as ML Service
    
    User->>UI: Uploads Dataset (CSV)
    UI->>API: POST /analyze (File)
    
    API->>ML: load_and_clean_data()
    Note over ML: Detects Target, Drops IDs, Imputes NaNs
    
    API->>ML: train_and_detect_fraud()
    
    Note over ML: SPLIT -> OVERSAMPLE (Fraud x N) -> TRAIN
    ML->>ML: Train XGBoost Model
    
    ML->>ML: Predict on Test Set
    ML->>ML: Dynamic Threshold Check (If 0 detected, lower thr)
    
    ML-->>API: Returns Metrics, Anomalies, Charts
    API-->>UI: JSON Response
    
    UI->>User: Displays Dashboard & Fraud Table
```

---

## 🧠 Model Details

### 1. XGBoost (Extreme Gradient Boosting)
We use XGBoost as the core classifier due to its superior performance on structured/tabular data. It is configured with:
*   `objective='binary:logistic'`: For binary classification (Fraud/Safe).
*   `scale_pos_weight`: To further penalize missing fraud cases.
*   `eval_metric='logloss'`: To optimize probabilistic output.

### 2. Oversampling Strategy
Credit card fraud datasets are highly imbalanced (e.g., 0.1% fraud). Standard models fail here (99.9% accuracy, 0% recall). 
**CreditGuard** solves this by:
1.  Splitting data into Train and Test.
2.  Isolating Fraud cases in the **Training Set**.
3.  **Duplicating (Oversampling)** these cases to achieve a balanced ratio (e.g., 50/50).
4.  Training the model on this "super-charged" dataset to learn subtle fraud patterns deeply.

---

## ⚙️ Setup & Installation

### Prerequisites
*   Python 3.8+
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/Abdullasaqib/CREDIT-CARD-FRAUD-DETECTION-USING-STATE-OF-THE-ART-MACHINE-LEARNING.git
cd crypto_fraud_detection
```

### 2. Create Virtual Environment

**For Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**For macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Application

1.  **Start the Server** (Ensure venv is active)
    ```bash
    uvicorn app.main:app --reload
    ```

2.  **Access the Dashboard**
    *   Open your browser and navigate to: `http://127.0.0.1:8000`
    *   **Upload** your dataset file.
    *   The system will process the file (2-10 seconds depending on size).
    *   View the **Detection Report** and scroll down for the **Fraud Data Table**.

---

## 📁 Project Structure

```
crypto_fraud_detection/
├── app/
│   ├── main.py              # Application Entry Point
│   ├── api.py               # API Route Handlers
│   ├── services/
│   │   ├── cleaning.py      # Preprocessing & Data Cleaning
│   │   └── modeling.py      # Core Machine Learning Logic
│   └── models.py            # Pydantic Schemas
├── static/
│   ├── index.html           # Main Dashboard UI
│   ├── style.css            # Cyber-Themed Styling
│   └── app.js               # Frontend Controller
├── uploads/                 # Temp Storage
├── venv/                    # Virtual Environment
└── requirements.txt         # Python Dependencies
```

---
**Developed for Final Year Project**