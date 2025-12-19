import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_clean_data(file_path, target_col_input=None):
    """
    Loads CSV, detects target column, and performs basic cleaning (Imputation, Encoding).
    Returns cleaned DataFrame, Target Column Name, and Dictionary of Encoders.
    """
    df = pd.read_csv(file_path)
    
    target_col = target_col_input
    
    if not target_col:
        # Auto-Detect Target Column
        possible_targets = ['isFraud', 'isfraud', 'fraud', 'class', 'target', 'label', 'Class', 'TARGET', 'IsFraud']
        
        # 1. Exact/Case-insensitive match
        for col in df.columns:
            if col in possible_targets:
                 target_col = col
                 break
            if col.lower() in [t.lower() for t in possible_targets]:
                target_col = col
                break
                
        # 2. Substring match
        if not target_col:
            for col in df.columns:
                if 'fraud' in col.lower() or 'class' in col.lower():
                    target_col = col
                    break
        
        # 3. Last Binary Column Fallback (Common in ML datasets)
        if not target_col:
            for col in reversed(df.columns):
                # Check if binary (0, 1) or (True, False)
                if df[col].nunique() == 2:
                    # Likely target
                    target_col = col
                    break

    # Data Cleaning
    # 1. Identify Meta Columns (IDs, Names) vs Features
    # We WANT to keep them for reporting, but DROP them for training
    
    # Heuristics for ID columns
    id_keywords = ['name', 'id', 'dest', 'orig', 'user', 'customer', 'account']
    # Explicitly drop technical artifacts that cause leakage (100% accuracy)
    drop_always = ['unnamed', 'index', 'step'] 
    
    meta_cols = []
    
    for col in df.columns:
        if col == target_col: continue
        
        col_lower = col.lower()
        if any(k in col_lower for k in drop_always):
            meta_cols.append(col)
            continue
            
        if df[col].dtype == 'object':
             # If high cardinality (many unique values), likely an ID
             if df[col].nunique() > 1000 or any(k in col_lower for k in id_keywords):
                 meta_cols.append(col)
                 
    # 2. Impute Missing Values
    # Numeric -> Mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude target_col from mean imputation to avoid introducing float values in classification targets
    feature_numeric_cols = [c for c in numeric_cols if c != target_col]
    
    if len(feature_numeric_cols) > 0:
        df[feature_numeric_cols] = df[feature_numeric_cols].fillna(df[feature_numeric_cols].mean())

    # Handle Target Column Distinctly
    # 1. Drop rows where Target is NaN (we can't train on them)
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
        
        # 2. Ensure Target is Integer (0/1) if it's numeric
        if pd.api.types.is_numeric_dtype(df[target_col]):
            df[target_col] = df[target_col].astype(int)
    
    # Categorical -> Mode (or 'Unknown')
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        if df[col].isnull().any():
             mode_val = df[col].mode()
             fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
             df[col] = df[col].fillna(fill_val)

    # 3. Encode Categorical Columns (Only for Features!)
    # We create a copy for training to avoid messing up the reporting DF
    df_train = df.copy()
    encoders = {}
    
    # effective_features are columns that are NOT meta_cols and NOT target
    effective_features = [c for c in df.columns if c not in meta_cols and c != target_col]
    
    for col in effective_features:
        if df_train[col].dtype == 'object':
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            encoders[col] = le
            
    # If target is object, encode it
    if df_train[target_col].dtype == 'object':
         le = LabelEncoder()
         df_train[target_col] = le.fit_transform(df_train[target_col].astype(str))
         encoders[target_col] = le

    # Drop meta cols from df_train
    df_train.drop(columns=meta_cols, inplace=True)
            
    # Return original DF (with IDs), Training DF, Target Col Name, Encoders, and list of Meta Cols
    return df, df_train, target_col, encoders, meta_cols
