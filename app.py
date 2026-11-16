import pickle
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained SVC model (saved as knnmodel.pkl)
with open("knnmodel.pkl", "rb") as file:
    model = pickle.load(file)

# Exact feature names and order from the pickle file
FEATURE_NAMES = ["Age", "Gender", "Region", "Occupation", "Income"]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Conversion Predictor | SVC Model</title>
    <style>
        :root {
            --bg-page: #EEF2FF;
            --bg-card: #FFFFFF;
            --primary-indigo: #1E1B4B;
            --accent-indigo: #4F46E5;
            --accent-hover: #4338CA;
            --text-main: #1E293B;
            --text-muted: #64748B;
            --border-color: #CBD5E1;
            --yes-bg: #ECFDF5;
            --yes-border: #10B981;
            --yes-text: #065F46;
            --no-bg: #FFF1F2;
            --no-border: #F43F5E;
            --no-text: #9F1239;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 640px;
            background: var(--bg-card);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(30, 27, 75, 0.1);
            overflow: hidden;
            border: 1px solid #E2E8F0;
        }

        .header {
            background: linear-gradient(135deg, var(--primary-indigo) 0%, #312E81 100%);
            color: #FFFFFF;
            padding: 2rem;
            border-bottom: 4px solid var(--accent-indigo);
        }

        .header h1 {
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
        }

        .header p {
            color: #C7D2FE;
            font-size: 0.92rem;
        }

        .content {
            padding: 2rem;
        }

        .result-box {
            padding: 1.25rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.75rem;
            border-left: 5px solid;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }

        .result-yes {
            background-color: var(--yes-bg);
            border-color: var(--yes-border);
            color: var(--yes-text);
        }

        .result-no {
            background-color: var(--no-bg);
            border-color: var(--no-border);
            color: var(--no-text);
        }

        .result-title {
            font-size: 1.15rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .result-badge {
            font-weight: 600;
            font-size: 0.85rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.8);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group.full-width {
            grid-column: 1 / -1;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--primary-indigo);
            margin-bottom: 0.4rem;
        }

        .hint {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 400;
            margin-left: 4px;
        }

        input, select {
            padding: 0.7rem 0.9rem;
            border: 1.5px solid var(--border-color);
            border-radius: 8px;
            font-size: 0.95rem;
            color: var(--text-main);
            background-color: #F8FAFC;
            transition: border-color 0.2s, box-shadow 0.2s, background-color 0.2s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent-indigo);
            background-color: #FFFFFF;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        button {
            width: 100%;
            margin-top: 1.75rem;
            background-color: var(--accent-indigo);
            color: #FFFFFF;
            border: none;
            padding: 0.95rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
        }

        button:hover {
            background-color: var(--accent-hover);
        }

        button:active {
            transform: scale(0.99);
        }

        .footer-note {
            margin-top: 1.5rem;
            text-align: center;
            font-size: 0.78rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Demographic Outcome Predictor</h1>
        <p>Enter applicant demographic and income details to classify outcome (Yes / No).</p>
    </div>

    <div class="content">
        {% if error %}
            <div class="result-box result-no">
                <div class="result-title">Error: {{ error }}</div>
            </div>
        {% endif %}

        {% if prediction is not none %}
            {% if prediction|lower == 'yes' %}
                <div class="result-box result-yes">
                    <div>
                        <div class="result-title">Prediction: YES</div>
                        <div style="font-size: 0.88rem; margin-top: 4px;">Positive class outcome predicted by SVC model.</div>
                    </div>
                    <div class="result-badge">Decision Score: {{ decision_score }}</div>
                </div>
            {% else %}
                <div class="result-box result-no">
                    <div>
                        <div class="result-title">Prediction: NO</div>
                        <div style="font-size: 0.88rem; margin-top: 4px;">Negative class outcome predicted by SVC model.</div>
                    </div>
                    <div class="result-badge">Decision Score: {{ decision_score }}</div>
                </div>
            {% endif %}
        {% endif %}

        <form method="POST" action="/">
            <div class="form-grid">
                <div class="form-group">
                    <label for="Age">Age <span class="hint">(years)</span></label>
                    <input type="number" id="Age" name="Age" min="1" max="120" required
                           value="{{ form_data.get('Age', '30') }}">
                </div>

                <div class="form-group">
                    <label for="Gender">Gender <span class="hint">(encoded)</span></label>
                    <select id="Gender" name="Gender" required>
                        <option value="0" {% if form_data.get('Gender', '0') == '0' %}selected{% endif %}>0 (Female / Class 0)</option>
                        <option value="1" {% if form_data.get('Gender') == '1' %}selected{% endif %}>1 (Male / Class 1)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="Region">Region <span class="hint">(encoded code)</span></label>
                    <select id="Region" name="Region" required>
                        <option value="0" {% if form_data.get('Region', '0') == '0' %}selected{% endif %}>Region 0</option>
                        <option value="1" {% if form_data.get('Region') == '1' %}selected{% endif %}>Region 1</option>
                        <option value="2" {% if form_data.get('Region') == '2' %}selected{% endif %}>Region 2</option>
                        <option value="3" {% if form_data.get('Region') == '3' %}selected{% endif %}>Region 3</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="Occupation">Occupation <span class="hint">(encoded code)</span></label>
                    <select id="Occupation" name="Occupation" required>
                        <option value="0" {% if form_data.get('Occupation', '0') == '0' %}selected{% endif %}>Occupation 0</option>
                        <option value="1" {% if form_data.get('Occupation') == '1' %}selected{% endif %}>Occupation 1</option>
                        <option value="2" {% if form_data.get('Occupation') == '2' %}selected{% endif %}>Occupation 2</option>
                        <option value="3" {% if form_data.get('Occupation') == '3' %}selected{% endif %}>Occupation 3</option>
                    </select>
                </div>

                <div class="form-group full-width">
                    <label for="Income">Income <span class="hint">(e.g., 25000 - 100000)</span></label>
                    <input type="number" step="any" id="Income" name="Income" min="0" required
                           value="{{ form_data.get('Income', '45000') }}">
                </div>
            </div>

            <button type="submit">Predict Outcome</button>
        </form>

        <div class="footer-note">
            Note: Ensure Gender, Region, and Occupation numeric codes match your training notebook's encoding.
        </div>
    </div>
</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    decision_score = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = request.form.to_dict()
            input_values = [
                float(request.form["Age"]),
                float(request.form["Gender"]),
                float(request.form["Region"]),
                float(request.form["Occupation"]),
                float(request.form["Income"]),
            ]

            input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)

            # Returns 'yes' or 'no'
            prediction = str(model.predict(input_df)[0])
            # Distance from the separating hyperplane (since probability=False in SVC)
            decision_score = round(float(model.decision_function(input_df)[0]), 3)

        except Exception as e:
            error = f"Invalid input or prediction error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        decision_score=decision_score,
        error=error,
        form_data=form_data,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
