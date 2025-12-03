"""
be_qc_lib_saved.py
Auto-generated safe library for predictions.
FIXED VERSION - works with both folder structures
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def load_target_models(out_dir, target):
    """
    Load models for a specific target.
    Handles both absolute paths (from local) and relative paths (for cloud).
    """
    # Try subdirectory structure first
    tdir = os.path.join(out_dir, target)
    meta_path = os.path.join(tdir, f"meta_{target}.json")
    
    # If subdirectory doesn't exist, try flat structure
    if not os.path.exists(meta_path):
        tdir = out_dir
        meta_path = os.path.join(out_dir, f"meta_{target}.json")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata for target '{target}' not found at: {meta_path}")
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # Smart path resolution: try to find files even if meta has absolute paths
    def resolve_model_path(meta_path_value, tdir, target, file_type):
        """Try multiple strategies to find the model file"""
        # Strategy 1: Use path as-is if it exists (for local absolute paths)
        if os.path.exists(meta_path_value):
            return meta_path_value
        
        # Strategy 2: Look for file by basename in target directory
        basename = os.path.basename(meta_path_value)
        candidate = os.path.join(tdir, basename)
        if os.path.exists(candidate):
            return candidate
        
        # Strategy 3: Try common naming patterns
        patterns = [
            f"{file_type}_{target}.joblib",
            f"{file_type}_{target}.pkl",
            f"lgb_{target}_{file_type}.txt",
            f"{file_type}_{target}.txt",
        ]
        for pattern in patterns:
            candidate = os.path.join(tdir, pattern)
            if os.path.exists(candidate):
                return candidate
        
        # If nothing works, raise error with helpful message
        raise FileNotFoundError(
            f"Could not find {file_type} file for {target}. "
            f"Looked in {tdir}. Available files: {os.listdir(tdir)}"
        )
    
    # Resolve paths for all model components
    try:
        preproc_path = resolve_model_path(meta["preproc"], tdir, target, "preproc")
        lgb_lower_path = resolve_model_path(meta["lgb_lower"], tdir, target, "lower")
        lgb_median_path = resolve_model_path(meta["lgb_median"], tdir, target, "median")
        lgb_upper_path = resolve_model_path(meta["lgb_upper"], tdir, target, "upper")
    except Exception as e:
        print(f"Error resolving paths for {target}: {str(e)}")
        raise
    
    # Load models
    preproc = joblib.load(preproc_path)
    m_low = lgb.Booster(model_file=lgb_lower_path)
    m_med = lgb.Booster(model_file=lgb_median_path)
    m_up  = lgb.Booster(model_file=lgb_upper_path)
    
    return preproc, m_low, m_med, m_up, meta

def predict_new(df_new, out_dir="be_qc_models", targets=None):
    if targets is None:
        targets = ["OUTPUT","INPUT","NILAI_DITAMBAH","GAJI_UPAH","JUMLAH_PEKERJA"]
    
    df = df_new.copy()
    
    # Derived features (only if columns present)
    if "JUMLAH_PEKERJA" in df.columns:
        df["log_workers"] = np.log1p(df["JUMLAH_PEKERJA"])
    if {"HARTA_TETAP","JUMLAH_PEKERJA"}.issubset(df.columns):
        df["asset_per_worker"] = (df["HARTA_TETAP"] / df["JUMLAH_PEKERJA"]).replace([np.inf, -np.inf], np.nan)
    if {"GAJI_UPAH","JUMLAH_PEKERJA"}.issubset(df.columns):
        df["wage_per_worker"] = (df["GAJI_UPAH"] / df["JUMLAH_PEKERJA"]).replace([np.inf, -np.inf], np.nan)
    
    df_out = df.copy()
    
    for t in targets:
        try:
            preproc, m_low, m_med, m_up, meta = load_target_models(out_dir, t)
        except Exception as e:
            # artifact missing; skip this target
            continue
        
        feats_num = meta.get("features_num", [])
        feats_cat = meta.get("features_cat", [])
        
        # Safe construction of X_raw
        X_raw = pd.DataFrame()
        for c in feats_num:
            if c in df_out.columns:
                X_raw[c] = to_num(df_out[c])
            else:
                X_raw[c] = np.nan
        for c in feats_cat:
            if c in df_out.columns:
                X_raw[c] = df_out[c].astype(str)
            else:
                X_raw[c] = "__MISS__"
        
        # Transform & predict
        X = preproc.transform(X_raw)
        df_out[f"{t}_PRED_MED"] = m_med.predict(X)
        df_out[f"{t}_PRED_LOW"] = m_low.predict(X)
        df_out[f"{t}_PRED_UP"]  = m_up.predict(X)
        
        # Flag if reported outside predicted interval (if reported exists)
        if t in df_out.columns:
            df_out[f"{t}_FLAG"] = ~df_out[t].between(df_out[f"{t}_PRED_LOW"], df_out[f"{t}_PRED_UP"])
            df_out.loc[df_out[t].isna(), f"{t}_FLAG"] = False
        else:
            df_out[f"{t}_FLAG"] = False
    
    # Composite flag
    flag_cols = [c for c in df_out.columns if c.endswith("_FLAG")]
    df_out["FLAG_FINAL"] = df_out[flag_cols].any(axis=1) if flag_cols else False
    
    return df_out

def predict_single(record_dict, out_dir="be_qc_models", targets=None):
    df = pd.DataFrame([record_dict])
    df_pred = predict_new(df, out_dir=out_dir, targets=targets)
    return df_pred.iloc[0].to_dict()
