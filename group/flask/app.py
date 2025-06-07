from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

FEATURE_NAMES = [
    'texture_mean','compactness_mean','concave points_mean','area_se',
    'concave points_se','texture_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave points_worst','symmetry_worst'
]


model = joblib.load('best_xgb_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    predictions = None

    if request.method == 'POST':
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
        else:
            try:
                user_input = [float(request.form.get(f, 0)) for f in FEATURE_NAMES]
                X = np.array(user_input).reshape(1, -1)
                pred = model.predict(X)[0]
                result = "Malignant (M)" if pred == 1 else "Benign (B)"
            except Exception as e:
                result = f"Input Error: {e}"

    return render_template('index.html', feature_names=FEATURE_NAMES, result=result, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
