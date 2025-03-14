from flask import Flask, jsonify
import joblib
import numpy as np
import os
from google.cloud import storage

app = Flask(__name__)

# Cloud Storage からモデルをロードする関数
def load_model():
    storage_client = storage.Client()
    bucket_name = os.getenv("MODEL_PATH").split("/")[2]  # `gs://credit-risk-bucket/credit_risk_model.pkl`
    blob_name = "/".join(os.getenv("MODEL_PATH").split("/")[3:])
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    local_model_path = "credit_risk_model.pkl"
    blob.download_to_filename(local_model_path)
    
    print("✅ モデルを Cloud Storage からダウンロードしました。")
    return joblib.load(local_model_path)

# モデルをロード
model = load_model()

@app.route("/", methods=["GET"])
def predict():
    """
    クレジットリスクを予測する
    """
    try:
        # サンプルデータ
        annual_income = 70000
        credit_score = 730
        loan_amount = 16000
        past_defaults = 0
        investment_frequency = 4

        # 入力データをモデルのフォーマットに変換
        features = np.array([[annual_income, credit_score, loan_amount, past_defaults, investment_frequency]])

        # 予測を実行
        prediction = model.predict_proba(features)[:, 1]

        return jsonify({"credit_risk_score": float(prediction[0])})  # ✅ JSON 形式で返す

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # ✅ エラー時も JSON を返す

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run は 8080 で実行
    app.run(host="0.0.0.0", port=port)