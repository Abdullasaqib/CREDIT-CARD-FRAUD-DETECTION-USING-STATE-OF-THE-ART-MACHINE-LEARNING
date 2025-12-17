import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import numpy as np

def train_and_detect_fraud(df_full, df_train, target_col, meta_cols):
    """
    Trains XGBoost with Hyperparameter Tuning.
    Generates Reporting data including Entity-level fraud analysis.
    """
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]
    
    # 1. Hyperparameter Tuning (Simplified for Speed)
    # We do a quick Randomized Search or just set robust params
    # For a real-time web app, full GridSearch is too slow.
    # We'll use a robust set of params and just fit.
    
    # Split Data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Also split the FULL dataframe so we can match rows later
        _, df_test_full, _, _ = train_test_split(df_full, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _, df_test_full, _, _ = train_test_split(df_full, y, test_size=0.2, random_state=42)
        
    
    # Handle Imbalance
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    # Define Model
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    
    # Simple Parameter Grid for "Tuning"
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0]
    }
    
    # Check if we have enough samples for CV
    min_class_samples = y_train.value_counts().min()
    if min_class_samples < 3:
        # Not enough samples for 3-fold CV, just fit the single model with default/mid params
        best_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            eval_metric='logloss'
        )
        best_model.fit(X_train, y_train)
    else:
        # Run Search
        # OPTIMIZATION: Use a subset for Hyperparameter Tuning if data is large
        # Training on 6M rows with CV=3 takes forever. We tune on 20k rows, then fit full model.
        if len(X_train) > 20000:
            # Stratified Sample
            try:
                X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=20000, stratify=y_train, random_state=42)
            except ValueError:
                 X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=20000, random_state=42)
        else:
            X_tune, y_tune = X_train, y_train

        random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=5, scoring='roc_auc', cv=3, random_state=42, n_jobs=1)
        random_search.fit(X_tune, y_tune)
        
        # Refit best model on FULL training set
        best_model = random_search.best_estimator_
        best_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Feature Importance
    importance = best_model.feature_importances_
    feat_imp = sorted(zip(X.columns, map(float, importance)), key=lambda x: x[1], reverse=True)
    
    # --- REPORTING GENERATION ---
    
    # 1. Attach Probabilities to Test Data
    df_test_full = df_test_full.copy()
    df_test_full['Predicted_Fraud_Prob'] = y_prob
    df_test_full['Actual_Fraud_Status'] = y_test
    
    # 2. Find "Amount" column (Heuristic)
    amount_col = None
    amount_keywords = ['amount', 'value', 'price', 'eth', 'usd', 'balance']
    for c in df_full.columns:
        if any(k in c.lower() for k in amount_keywords) and c not in meta_cols:
            amount_col = c
            break
            
    # 3. Find "Entity" column (Heuristic: nameOrig, account_id, etc.)
    entity_col = None
    possible_entities = ['nameOrig', 'customer', 'account', 'user', 'name', 'id']
    for c in df_full.columns:
        if any(k in c.lower() for k in possible_entities):
            entity_col = c
            break
            
    # 4. Filter High Risk Transactions
    # Threshold > 0.5 (since we optimized for imbalance, 0.5 is fair, or use 0.7)
    high_risk_df = df_test_full[df_test_full['Predicted_Fraud_Prob'] > 0.5]
    
    # 5. Generate PIE CHART Data (Total Amount by Entity)
    pie_data = []
    if amount_col and entity_col and not high_risk_df.empty:
        # Group by Entity and Sum Amount
        fraud_by_entity = high_risk_df.groupby(entity_col)[amount_col].sum().sort_values(ascending=False)
        
        # Take Top 5 and group rest as 'Others'
        top_5 = fraud_by_entity.head(5)
        others_sum = fraud_by_entity.iloc[5:].sum() if len(fraud_by_entity) > 5 else 0
        
        for name, amount in top_5.items():
            pie_data.append({"label": str(name), "value": float(amount)})
            
        if others_sum > 0:
            pie_data.append({"label": "Others", "value": float(others_sum)})
            
    elif not high_risk_df.empty:
        # Fallback if no specific Amount column: Count of Fraud Txs by Entity
         if entity_col:
             fraud_by_entity = high_risk_df[entity_col].value_counts().head(5)
             for name, count in fraud_by_entity.items():
                 pie_data.append({"label": str(name), "value": int(count)})
         else:
             # Just show Verified vs Fraud Count
             pie_data.append({"label": "Fraud Transactions", "value": len(high_risk_df)})
             pie_data.append({"label": "Safe Transactions", "value": len(df_test_full) - len(high_risk_df)})

    # 6. Top Anomalies List
    top_anomalies = high_risk_df.sort_values(by='Predicted_Fraud_Prob', ascending=False).head(20)
    top_anomalies = top_anomalies.fillna("N/A")
    anomalies_list = top_anomalies.to_dict(orient='records')

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)) if len(set(y_test)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    return {
        "metrics": metrics,
        "feature_importance": feat_imp,
        "anomalies": anomalies_list,
        "pie_data": pie_data,
        "entity_col": entity_col,
        "amount_col": amount_col
    }
