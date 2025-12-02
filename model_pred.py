import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

# OCR imports
from PIL import Image
import easyocr
import io
import base64

app = Flask(__name__)
CORS(app)

# ================================
# LOAD OCR MODEL
# ================================
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True)
print("EasyOCR Loaded!")

# ================================
# LOAD ML MODELS & TRAIN COLUMNS
# ================================
print("Loading ML Models...")

models = {
    "pred_7d": joblib.load("pred_7d.pkl"),
    "pred_30d": joblib.load("pred_30d.pkl"),
    "pred_60d": joblib.load("pred_60d.pkl"),
    "pred_180d": joblib.load("pred_180d.pkl"),
}

train_columns = joblib.load("train_columns.pkl")

print("ML Models Loaded!")



# ==========================================================
# ===================== OCR ROUTE ==========================
# ==========================================================
@app.route("/extract", methods=["POST"])
def extract_text():
    files = request.files.getlist("image")
    
    if not files:
        return jsonify({"error": "No image provided"}), 400

    all_extracted_text = []

    try:
        for file in files:
            image = Image.open(file.stream)

            # OCR extraction
            img_array = np.array(image)
            results = reader.readtext(img_array, detail=0)
            text = "\n".join(results)
            all_extracted_text.append(text)

        combined_text = "\n\n--- Next Image ---\n\n".join(all_extracted_text)

        return jsonify({
            "success": True,
            "extracted_text": combined_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# =================== ORIGINAL ML ROUTE ====================
# ==========================================================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Ensure list input
    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    # 1️⃣ Date → days_since_restock
    if "last_restock_date" in df.columns:
        df["last_restock_date"] = pd.to_datetime(df["last_restock_date"], errors="coerce")
        today = pd.Timestamp.today()
        df["days_since_restock"] = (today - df["last_restock_date"]).dt.days
        df = df.drop(columns=["last_restock_date"])

    print("completed1")

    # 2️⃣ Fill missing numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    print("completed2")

    # 3️⃣ Encode categories
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print("completed3")

    # 4️⃣ Align to training columns
    df = df.reindex(columns=train_columns, fill_value=0)

    print("completed4")

    # ==============================================
    # 5️⃣ VECTORIZED PREDICTION (FAST)
    # ==============================================
    pred_7  = models["pred_7d"].predict(df)
    pred_30 = models["pred_30d"].predict(df)
    pred_60 = models["pred_60d"].predict(df)
    pred_180 = models["pred_180d"].predict(df)

    print("completed5")

    # ==============================================
    # 6️⃣ RESTOCK LOGIC (vectorized)
    # ==============================================
    stock = np.array([item.get("stock_level", 0) for item in data])
    
    # Try to get daily_demand from input, otherwise calculate from predictions
    input_daily_demand = np.array([item.get("daily_demand", 0) for item in data])
    
    # Calculate derived daily demand from 30-day prediction
    derived_daily_demand = pred_30 / 30.0
    
    # Use input daily_demand if available (>0), otherwise use derived
    daily_demand = np.where(input_daily_demand > 0, input_daily_demand, derived_daily_demand)

    # Avoid division by zero
    days_left = np.where(daily_demand <= 0, 9999, stock / daily_demand)

    need_7  = pred_7  > stock
    need_30 = pred_30 > stock
    need_60 = pred_60 > stock
    need_180 = pred_180 > stock

    qty_7  = np.maximum(pred_7  - stock, 0)
    qty_30 = np.maximum(pred_30 - stock, 0)
    qty_60 = np.maximum(pred_60 - stock, 0)
    qty_180 = np.maximum(pred_180 - stock, 0)

    print("completed6")

    # ==============================================
    # 7️⃣ BUILD RESPONSE
    # ==============================================
    predictions = []

    for i, item in enumerate(data):
        predictions.append({
            "item_id": item.get("item_id"),
            "stock":float(stock[i]),

            "pred_7d": int(round(pred_7[i])),
            "pred_30d": int(round(pred_30[i])),
            "pred_60d": int(round(pred_60[i])),
            "pred_180d": int(round(pred_180[i])),
            "days_left": int(round(days_left[i])),

            "need_restock_7d": bool(need_7[i]),
            "need_restock_30d": bool(need_30[i]),
            "need_restock_60d": bool(need_60[i]),
            "need_restock_180d": bool(need_180[i]),

            "restock_qty_7d": float(qty_7[i]),
            "restock_qty_30d": float(qty_30[i]),
            "restock_qty_60d": float(qty_60[i]),
            "restock_qty_180d": float(qty_180[i]),
        })

    print("completed7")

    return jsonify(predictions)


# ==========================================================
# ==================== START SERVER ========================
# ==========================================================
if __name__ == "__main__":
    print("\n🚀 Unified Backend (OCR + ML) running at http://localhost:5000")
    print("➡ /extract → OCR")
    print("➡ /predict → ML Prediction\n")

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
