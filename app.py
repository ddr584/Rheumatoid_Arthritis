import os
import pandas as pd
import pickle
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load model or train if not available
try:
    with open('model/trained_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)
except FileNotFoundError:
    data = pd.read_excel('ra_dataset_500.xlsx')
    X = data[['age', 'CRP_levels', 'ESR', 'joint_pain_score', 'fatigue_score']]
    y = data['RA_severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trained_model = RandomForestClassifier(n_estimators=100, random_state=42)
    trained_model.fit(X_train, y_train)

    os.makedirs('model', exist_ok=True)
    with open('model/trained_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            float(request.form['CRP_levels']),
            float(request.form['ESR']),
            float(request.form['joint_pain_score']),
            float(request.form['fatigue_score'])
        ]
        prediction = trained_model.predict([features])[0]
        return render_template('index.html', prediction_text=f'Predicted RA Severity: {prediction}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))