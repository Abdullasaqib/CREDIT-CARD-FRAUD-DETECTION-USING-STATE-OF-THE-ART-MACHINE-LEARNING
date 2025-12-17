from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import shutil
import os
import pandas as pd
from app.services.cleaning import load_and_clean_data
from app.services.modeling import train_and_detect_fraud

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    """
    Single Step: Upload -> Auto-Detect Target -> Analyze
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save Uploaded File
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Step 1: Cleaning & Processing (Auto-Detect Target)
        # Returns: full_df, training_df, target_col, encoders, meta_cols
        df_full, df_train, target_col, encoders, meta_cols = load_and_clean_data(file_path)
        
        if not target_col:
             raise HTTPException(status_code=400, detail="Could not automatically detect a Target/Label column. Please check your dataset.")

        # Step 2: Modeling (XGBoost)
        result = train_and_detect_fraud(df_full, df_train, target_col, meta_cols)
        
        return {
            "status": "success",
            "filename": file.filename,
            "target_detected": target_col,
            "results": result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
