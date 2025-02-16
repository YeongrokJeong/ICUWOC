import joblib
import numpy as np
import pandas as pd
import shap
import warnings
import requests
import os

warnings.filterwarnings("ignore", category=UserWarning, module='shap')
while True:
    try:
        feature1 = float(input("의식수준을 입력하세요 (-5 ~ +4): "))
        if -5 <= feature1 <= 4:
            break
        else:
            print("의식수준은 -5에서 +4 사이의 값이어야 합니다.")
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해 주세요.")

while True:
    try:
        feature2 = float(input("체온을 입력하세요: "))
        break
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해 주세요.")

while True:
    try:
        feature3 = float(input("실금을 입력하세요: "))
        break
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해 주세요.")

while True:
    try:
        feature4 = float(input("좌우 하지근력의 합을 입력하세요 (0 ~ 10): "))
        if 0 <= feature4 <= 10:
            break
        else:
            print("하지근력은 0에서 10 사이의 값이어야 합니다.")
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해 주세요.")

# GitHub에서 모델 파일 다운로드
url = "https://raw.githubusercontent.com/YeongrokJeong/ICUWOC/main/ulcer_prediction_0216.pkl"
filename = "ulcer_prediction_0216.pkl"

# URL에서 파일 다운로드
response = requests.get(url)
with open(filename, "wb") as f:
    f.write(response.content)

# 다운로드한 파일을 joblib으로 로드
loaded_model = joblib.load(filename)

# SHAP 분석기 준비
test_sample = pd.read_csv('https://raw.githubusercontent.com/YeongrokJeong/ICUWOC/main/shap_value_0216.csv')
explainer = shap.KernelExplainer(loaded_model.predict_proba, test_sample)

new_data = pd.DataFrame([[feature1, feature2, feature3, feature4]])
proba=loaded_model.predict_proba(new_data)[0,1]

shap_values = explainer.shap_values(new_data)
shap_df = pd.DataFrame({
    "Feature": ['의식수준','실금','하지근력','체온'],
    "SHAP Value": np.round(shap_values[0][:, 1] * 100, 1),  # Class 1에 대한 SHAP 값
    "Feature Value": new_data.values[0]  # 해당 데이터 포인트의 특성 값
})

positive_shap_df = shap_df[shap_df['SHAP Value'] > 0]
print(f"욕창 발생 확률은 {round(proba * 100, 1)}%입니다.")

for index, row in positive_shap_df.iterrows():
    print(f"'{row['Feature']}'이 욕창 발생 확률 증가에 {row['SHAP Value']}%만큼 기여하였습니다.")
