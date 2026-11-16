import os
import pickle
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained SVM model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Model Predictor</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: #1e293b;
            --card-border: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --input-bg: #0f172a;
            --yes-bg: rgba(16, 185, 129, 0.15);
            --yes-border: #10b981;
            --yes-text: #34d399;
            --no-bg: rgba(244, 63, 94, 0.15);
            --no-border: #f43f5e;
            --no-text: #fb7185;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
            width: 100%;
            max-width: 480px;
            padding: 32px;
        }

        .header {
            text-align: center;
            margin-bottom: 28px;
        }

        .header h1 {
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 6px;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .form-group {
            margin-bottom: 18px;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        input, select {
            width: 100%;
            padding: 12px 14px;
            background-color: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 0.95rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        .row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        button {
            width: 100%;
            padding: 13px;
            margin-top: 8px;
            background-color: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
        }

        button:hover {
            background-color: var(--accent-hover);
        }

        button:active {
            transform: scale(0.99);
        }

        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            font-size: 1.05rem;
            animation: fadeIn 0.3s ease-in-out;
        }

        .result-yes {
            background-color: var(--yes-bg);
            border: 1px solid var(--yes-border);
            color: var(--yes-text);
        }

        .result-no {
            background-color: var(--no-bg);
            border: 1px solid var(--no-border);
            color: var(--no-text);
        }

        .result-error {
            background-color: rgba(245, 158, 11, 0.15);
            border: 1px solid #f59e0b;
            color: #fbbf24;
            font-size: 0.9rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SVM Classification Portal</h1>
            <p>Enter applicant features to run RBF kernel inference</p>
        </div>

        <form method="POST" action="/">
            <div class="row">
                <div class="form-group">
                    <label for="Age">Age</label>
                    <input type="number" step="any" id="Age" name="Age" placeholder="e.g., 32" value="{{ form_data.get('Age', '') }}" required>
                </div>

                <div class="form-group">
                    <label for="Gender">Gender (Encoded)</label>
                    <select id="Gender" name="Gender" required>
                        <option value="0" {% if form_data.get('Gender') == '0' %}selected{% endif %}>0 (Female)</option>
                        <option value="1" {% if form_data.get('Gender') == '1' %}selected{% endif %}>1 (Male)</option>
                    </select>
                </div>
            </div>

            <div class="row">
                <div class="form-group">
                    <label for="Region">Region (Encoded)</label>
                    <input type="number" step="any" id="Region" name="Region" placeholder="e.g., 0, 1, 2" value="{{ form_data.get('Region', '') }}" required>
                </div>

                <div class="form-group">
                    <label for="Occupation">Occupation (Encoded)</label>
                    <input type="number" step="any" id="Occupation" name="Occupation" placeholder="e.g., 0, 1, 2" value="{{ form_data.get('Occupation', '') }}" required>
                </div>
            </div>

            <div class="form-group">
                <label for="Income">Income</label>
                <input type="number" step="any" id="Income" name="Income" placeholder="e.g., 55000" value="{{ form_data.get('Income', '') }}" required>
            </div>

            <button type="submit">Predict Outcome</button>
        </form>

        {% if prediction %}
            <div class="result-box {% if prediction|lower == 'yes' %}result-yes{% else %}result-no{% endif %}">
                Model Prediction: {{ prediction|upper }}
            </div>
        {% endif %}

        {% if error %}
            <div class="result-box result-error">
                Error: {{ error }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            # Build DataFrame with exact column names matching feature_names_in_
            input_df = pd.DataFrame([{
                "Age": float(request.form["Age"]),
                "Gender": float(request.form["Gender"]),
                "Region": float(request.form["Region"]),
                "Occupation": float(request.form["Occupation"]),
                "Income": float(request.form["Income"]),
            }])

            pred = model.predict(input_df)[0]
            prediction = str(pred)
        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        error=error,
        form_data=form_data
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
