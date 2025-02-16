from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import shap
import warnings
import requests
import os

warnings.filterwarnings("ignore", category=UserWarning, module='shap')

app = Flask(__name__)

# GitHub에서 모델 다운로드 (raw URL 사용)
PKL_URL = "https://raw.githubusercontent.com/YeongrokJeong/ICUWOC/main/ulcer_prediction_0216.pkl"
PKL_FILE = "ulcer_prediction_0216_test.pkl"

def download_and_load_model():
    """GitHub에서 모델 다운로드 후 로드"""
    if not os.path.exists(PKL_FILE):  # 모델이 없으면 다운로드
        response = requests.get(PKL_URL)
        with open(PKL_FILE, "wb") as f:
            f.write(response.content)
    return joblib.load(PKL_FILE)

# Flask가 실행될 때 모델을 미리 로드
model = download_and_load_model()

# SHAP 분석기 준비
test_sample = pd.read_csv('https://raw.githubusercontent.com/YeongrokJeong/ICUWOC/main/shap_value_0216.csv')
explainer = shap.KernelExplainer(model.predict_proba, test_sample)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 사용자가 입력한 값 가져오기
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])

        # 입력값 검증
        if not (-5 <= feature1 <= 4):
            return jsonify({"error": "의식수준은 -5에서 +4 사이여야 합니다."})
        if not (0 <= feature3 <= 10):
            return jsonify({"error": "하지근력은 0에서 10 사이여야 합니다."})

        # 모델 입력 데이터 생성
        new_data = pd.DataFrame([[feature1, feature2, feature3, feature4]])
        proba = model.predict_proba(new_data)[0, 1]

        # SHAP 값 계산
        shap_values = explainer.shap_values(new_data)
        shap_df = pd.DataFrame({
            "Feature": ['의식수준', '실금횟수', '하지근력', '체온'],
            "SHAP Value": np.round(shap_values[0][:, 1] * 100, 1),
            "Feature Value": new_data.values[0]
        })

        shap_df.iloc[:,2]=shap_df.iloc[:,2]*100

        positive_shap_df = shap_df[shap_df['SHAP Value'] > 0]
    

        # 결과 데이터 정리
        shap_results = positive_shap_df.to_dict(orient='records')

        return jsonify({
            "probability": round(proba * 100, 1),
            "shap_values": shap_results
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render에서는 환경 변수 PORT 사용
    app.run(host="0.0.0.0", port=port)
