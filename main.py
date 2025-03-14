from flask import Flask, jsonify, request
import joblib
import numpy as np
import os
from google.cloud import storage

app = Flask(__name__)

# Cloud Storage からモデルをロードする関数
def load_model():
    storage_client = storage.Client()
    bucket_name = os.getenv("MODEL_PATH").split("/")[2]  # `gs://credit-risk-bucket-test/credit_risk_model.pkl`
    blob_name = "/".join(os.getenv("MODEL_PATH").split("/")[3:])
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    local_model_path = "credit_risk_model.pkl"
    blob.download_to_filename(local_model_path)
    
    print("✅ モデルを Cloud Storage からダウンロードしました。")
    return joblib.load(local_model_path)

# モデルをロード
model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    """
    クレジットリスクを予測するAPI
    - `annual_income`: 年収（float）
    - `credit_score`: クレジットスコア（int）
    - `loan_amount`: ローン金額（float）
    - `past_defaults`: 過去のデフォルト回数（int）
    - `investment_frequency`: 投資頻度（int）
    """
    try:
        # JSONデータを取得
        data = request.get_json()

        # 必須のキーがあるかチェック
        required_keys = ["annual_income", "credit_score", "loan_amount", "past_defaults", "investment_frequency"]
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required parameters"}), 400

        # 入力データをモデルのフォーマットに変換
        features = np.array([[data["annual_income"], data["credit_score"], data["loan_amount"], 
                              data["past_defaults"], data["investment_frequency"]]])

        # 予測を実行
        prediction = model.predict_proba(features)[:, 1]

        return jsonify({"credit_risk_score": float(prediction[0])})  # ✅ JSON 形式で返す

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # ✅ エラー時も JSON を返す

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)