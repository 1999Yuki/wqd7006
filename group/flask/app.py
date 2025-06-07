from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

FEATURE_NAMES = [
    'texture_mean', 'compactness_mean', 'concave points_mean', 'area_se',
    'concave points_se', 'texture_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst'
]

# 默认 Benign (B) 样本
DEFAULT_USER_INPUT = {
    'texture_mean': 14.36,
    'compactness_mean': 0.08129,
    'concave points_mean': 0.04781,
    'area_se': 23.56,
    'concave points_se': 0.01315,
    'texture_worst': 19.26,
    'area_worst': 711.2,
    'smoothness_worst': 0.144,
    'compactness_worst': 0.1773,
    'concavity_worst': 0.239,
    'concave points_worst': 0.1288,
    'symmetry_worst': 0.2977
}

model = joblib.load('best_xgb_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    predictions = None
    user_input = DEFAULT_USER_INPUT.copy()  # 默认用 Benign 样本

    if request.method == 'POST':
        if 'excel_file' in request.files and request.files['excel_file'].filename != '':
            file = request.files['excel_file']
            if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                df = pd.read_excel(file)
                try:
                    X_batch = df[FEATURE_NAMES].values
                    preds = model.predict(X_batch)
                    predictions = [
                        (dict(zip(FEATURE_NAMES, row)), "Malignant (M)" if p == 1 else "Benign (B)")
                        for row, p in zip(X_batch, preds)
                    ]
                except Exception as e:
                    result = f"Excel Processing Error: {e}"
        else:
            try:
                user_input = {f: request.form.get(f, '') for f in FEATURE_NAMES}
                input_list = [float(user_input[f]) if user_input[f] != '' else 0 for f in FEATURE_NAMES]
                X = np.array(input_list).reshape(1, -1)
                pred = model.predict(X)[0]
                result = "Malignant (M)" if pred == 1 else "Benign (B)"
            except Exception as e:
                result = f"Input Error: {e}"

    return render_template(
        'index.html',
        feature_names=FEATURE_NAMES,
        user_input=user_input,
        result=result,
        predictions=predictions
    )

if __name__ == '__main__':
    app.run(debug=True)
