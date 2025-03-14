#①bigquery接続　データ取得
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

query = """
SELECT annual_income, credit_score, loan_amount, past_defaults, investment_frequency, default_next_6_months
FROM credit_risk_dataset.investors
"""
df = client.query(query).to_dataframe()
df.head()

#②モデル作成
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# データの前処理
X = df.drop(columns=["default_next_6_months"])
y = df["default_next_6_months"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# モデルの学習
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 評価
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"AUC-ROC Score: {auc:.3f}")


#③Cloud Storage接続確認
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('credit-risk-bucket-test')
blobs = client.list_blobs(bucket)
for blob in blobs:
    print(blob.name)


#④Cloud Storageにモデルをアップロード
import joblib
from google.cloud import storage

# モデルを保存
model_filename = "credit_risk_model2.pkl"
joblib.dump(model, model_filename)

# Cloud Storage にアップロード
bucket_name = "credit-risk-bucket-test" #bucketがないとエラーになる
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(model_filename)
blob.upload_from_filename(model_filename)