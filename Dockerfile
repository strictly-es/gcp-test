# Python の軽量バージョンを使用
FROM python:3.9-slim

# 作業ディレクトリ作成
WORKDIR /app

# 必要なファイルをコピー
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー
COPY . .

# ポートを開放
EXPOSE 8080

# アプリを実行
CMD ["python", "main.py"]