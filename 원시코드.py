import joblib
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='shap')

# 의식수준 입력 (범위 -2 ~ +2)
while True:
    try:
        feature1 = float(input("의식수준을 입력하세요 (-5 ~ +4): "))
        if -5 <= feature1 <= 4:
            break
        else:
            print("의식수준은 -5에서 +4 사이의 값이어야 합니다.")
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해 주세요.")

# 체온 입력 (범위 제한 없음)
while True:
    try:
        feature2 = float(input("체온을 입력하세요: "))
        break
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해 주세요.")

# 실금 입력 (범위 제한 없음)
while True:
    try:
        feature3 = float(input("실금을 입력하세요: "))
        break
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해 주세요.")

# 하지근력 입력 (범위 0 ~ 10)
while True:
    try:
        feature4 = float(input("좌우 하지근력의 합을 입력하세요 (0 ~ 10): "))
        if 0 <= feature4 <= 10:
            break
        else:
            print("하지근력은 0에서 10 사이의 값이어야 합니다.")
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해 주세요.")

# 중환자실 재원기간 입력 (범위 제한 없음)
while True:
    try:
        feature5 = float(input("중환자실 재원기간을 입력하세요: "))
        break
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해 주세요.")

loaded_model = joblib.load(r'C:\Users\정영록\Desktop\PythonWorkspace\ulcer_prediction_project\ulcer_prediction.pkl')

new_data = pd.DataFrame([[feature1, feature2, feature3/feature5, feature4]])
proba=loaded_model.predict_proba(new_data)[0,1]

shap_value=pd.read_csv(r'C:\Users\정영록\Desktop\PythonWorkspace\ulcer_prediction_project\shap_value.csv')
test_sample = shap_value.sample(999, random_state=42)
explainer = shap.KernelExplainer(loaded_model.predict_proba, test_sample)

shap_values = explainer.shap_values(new_data)
shap_df = pd.DataFrame({
    "Feature": shap_value.columns,
    "SHAP Value": np.round(shap_values[0][:, 1] * 100, 1),  # Class 1에 대한 SHAP 값
    "Feature Value": new_data.values[0]  # 해당 데이터 포인트의 특성 값
})

positive_shap_df = shap_df[shap_df['SHAP Value'] > 0]
print(f"욕창 발생 확률은 {round(proba * 100, 1)}%입니다.")

for index, row in positive_shap_df.iterrows():
    print(f"'{row['Feature']}'이 욕창 발생 확률 증가에 {row['SHAP Value']}%만큼 기여하였습니다.")