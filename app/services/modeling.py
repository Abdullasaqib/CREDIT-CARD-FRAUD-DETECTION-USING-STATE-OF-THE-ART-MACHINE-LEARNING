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
    
    # OVERSAMPLING LOGIC: Explicitly duplicate fraud cases in the TRAINING set
    # We do this AFTER the initial split to avoid training on test data (leakage), 
    # BUT the complex logic below does split first. 
    # Let's clean up the split logic to handle oversampling properly on X_train/y_train ONLY.
    
    # 1. Initial Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Also split the FULL dataframe so we can match rows later
        _, df_test_full, _, _ = train_test_split(df_full, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _, df_test_full, _, _ = train_test_split(df_full, y, test_size=0.2, random_state=42)
        
    # 2. Oversample Training Data Only
    # Combine X_train and y_train temporarily
    train_data = X_train.copy()
    train_data['__target__'] = y_train
    
    fraud_data = train_data[train_data['__target__'] == 1]
    safe_data = train_data[train_data['__target__'] == 0]
    
    # Target: 50/50 ratio or at least boost fraud count significantly
    if len(fraud_data) > 0 and len(safe_data) > 0:
        # Calculate how many to sample. If fraud is very small, duplicate it many times.
        # Let's aim for count(safe) // 2 (33% fraud) or even count(safe) (50% fraud)
        n_samples = len(safe_data)
        
        fraud_oversampled = fraud_data.sample(n_samples, replace=True, random_state=42)
        train_oversampled = pd.concat([safe_data, fraud_oversampled])
        
        # Shuffle
        train_oversampled = train_oversampled.sample(frac=1, random_state=42)
        
        # Splot back
        X_train = train_oversampled.drop(columns=['__target__'])
        y_train = train_oversampled['__target__']
        
    
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
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    # Dynamic Threshold Adjustment
    # If default threshold (0.5) yields 0 positives but we suspect there's fraud, try lower thresholds.
    if y_pred.sum() == 0:
         for thr in [0.4, 0.3, 0.2, 0.1, 0.05]:
             y_pred_adj = (y_prob > thr).astype(int)
             if y_pred_adj.sum() > 0:
                 y_pred = y_pred_adj
                 # Also update the High Risk filter threshold implicitly by using y_pred logic later if desired,
                 # but for now we just fix the Metrics reported.
                 break
    
    # Feature Importance
    importance = best_model.feature_importances_
    feat_imp = sorted(zip(X.columns, map(float, importance)), key=lambda x: x[1], reverse=True)
    
    # --- REPORTING GENERATION ---
    
    # 1. Attach Probabilities AND Predictions to Test Data
    df_test_full = df_test_full.copy()
    df_test_full['Predicted_Fraud_Prob'] = y_prob
    df_test_full['Predicted_Class'] = y_pred
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
    # STRICTLY use the final Model Prediction (y_pred), which includes any dynamic threshold adjustments.
    # The user wants "only rows predicted as fraud".
    high_risk_df = df_test_full[df_test_full['Predicted_Class'] == 1]
    
    # If NO fraud is predicted strictly, we technically should return empty for "Fraud Rows",
    # but to show *something* in the pie chart we might need logic. 
    # However, for the LIST (anomalies), we should obey "only predicted as fraud".
    
    # Pie Data Fallback Logic needs to remain robust if high_risk_df is empty.
    
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

    elif amount_col and not high_risk_df.empty:
         # Fallback: No Entity Name, but we have Amount. Show Top 5 Transactions by Amount (Txn ID)
         # Sort by Amount
         top_by_amount = high_risk_df.sort_values(by=amount_col, ascending=False).head(5)
         
         for idx, row in top_by_amount.iterrows():
             pie_data.append({"label": f"Txn #{idx}", "value": float(row[amount_col])})
             
         # Note: We won't show 'Others' here because it makes less sense for individual transactions
         # unless we want to sum the rest. Let's just show top 5 high value frauds.
            
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
             
    else:
        # No High Risk Transactions Found
        # Show breakdown of Safe vs Fraud (which is 0) to avoid empty chart
        pie_data.append({"label": "No Fraud Detected", "value": 100})
        # Or showing the full count of safe transactions
        # pie_data.append({"label": "Safe Transactions", "value": len(df_test_full)})

    # 6. All High Risk Anomalies
    top_anomalies = high_risk_df.sort_values(by='Predicted_Fraud_Prob', ascending=False)
    # Using head(N) limits us, but returning 1M rows crashes browser.
    # User asked for "all fraud rows". Let's cap at a reasonable large number or just give all.
    # If list is > 5000, maybe we should warn, but user said "everything". 
    # We will return ALL rows.
    top_anomalies = top_anomalies
    top_anomalies = top_anomalies.fillna("N/A")
    anomalies_list = top_anomalies.to_dict(orient='records')

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)) if len(set(y_test)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    }

    return {
        "metrics": metrics,
        "feature_importance": feat_imp,
        "anomalies": anomalies_list,
        "pie_data": pie_data,
        "entity_col": entity_col,
        "amount_col": amount_col
    }
