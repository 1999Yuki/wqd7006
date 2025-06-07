from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

FEATURE_NAMES = [
    'texture_mean','compactness_mean','concave points_mean','area_se',
    'concave points_se','texture_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave points_worst','symmetry_worst'
]

# 默认预填的特征值（来自你提供的图片）
DEFAULT_VALUES = [
    14.36, 0.08129, 0.04781, 23.56, 0.01315, 19.26,
    711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977
]

model = joblib.load('best_xgb_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    predictions = None
    input_values = DEFAULT_VALUES.copy()

    if request.method == 'POST':
        # 文件上传（Excel 批量预测）
        if 'excel_file' in request.files:
            file = request.files['excel_file']
            if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                df = pd.read_excel(file)
                try:
                    X_batch = df[FEATURE_NAMES].values
                    preds = model.predict(X_batch)
                    predictions = [(row, "Malignant (M)" if p == 1 else "Benign (B)") for row, p in zip(X_batch, preds)]
                except Exception as e:
                    result = f"Excel Processing Error: {e}"
            return render_template('index.html', feature_names=FEATURE_NAMES, input_values=input_values, predictions=predictions, result=None)

        # 单条输入预测（AJAX 返回 JSON）
        else:
            try:
                input_values = [float(request.form.get(f, 0)) for f in FEATURE_NAMES]
                X = np.array(input_values).reshape(1, -1)
                pred = model.predict(X)[0]
                result = "Malignant (M)" if pred == 1 else "Benign (B)"
                return jsonify({'result': result})
            except Exception as e:
                return jsonify({'result': f"Input Error: {e}"})

    return render_template('index.html', feature_names=FEATURE_NAMES, input_values=input_values, predictions=predictions, result=result)

if __name__ == '__main__':
    app.run(debug=True)
