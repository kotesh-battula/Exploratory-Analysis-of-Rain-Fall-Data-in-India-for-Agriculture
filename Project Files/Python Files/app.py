# app.py (No changes needed, loads the new robust pipeline)
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration & Model Loading ---
MODEL_PATH = 'Rainfall_pipeline.pkl' 
pipeline = None
FEATURE_COLUMNS = []

try:
    model_data = joblib.load(MODEL_PATH)
    pipeline = model_data['pipeline']
    FEATURE_COLUMNS = model_data['feature_columns']
    print("Model and metadata loaded successfully.")
except Exception as e:
    print(f"ERROR: Model not loaded. Check server logs: {e}")
    print("Please ensure you have run train_model.py to create Rainfall_pipeline.pkl.")

# Define expected data types and defaults for robustness
NUMERIC_FEATURES = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 
    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
    'Temp9am', 'Temp3pm', 'Day', 'Month'
]
CATEGORICAL_FEATURES = [
    'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'
]

def preprocess_input(form_data):
    """
    Converts form data into a DataFrame suitable for the model pipeline, 
    handling missing features or invalid numeric inputs robustly to prevent KeyError.
    """
    input_dict = {}

    # 1. Handle numeric features
    for col in NUMERIC_FEATURES:
        val = form_data.get(col)
        try:
            input_dict[col] = [float(val)]
        except (ValueError, TypeError):
            input_dict[col] = [np.nan]

    # 2. Handle categorical features
    for col in CATEGORICAL_FEATURES:
        input_dict[col] = [form_data.get(col, '')]

    # 3. Handle 'RainToday' feature (1 for 'Yes', 0 otherwise)
    rain_today_str = form_data.get('RainToday')
    input_dict['RainToday'] = [1 if rain_today_str == 'Yes' else 0]

    # 4. Create DataFrame in the exact feature order
    X_new = pd.DataFrame(input_dict)[FEATURE_COLUMNS]
    
    return X_new

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if pipeline is None:
            return render_template('noChance.html', message="Model not loaded. Ensure Rainfall_pipeline.pkl exists.")

        try:
            X_new = preprocess_input(request.form)
            
            # The pipeline uses the correct feature names saved during training
            prediction_proba = pipeline.predict_proba(X_new)[0, 1] 
            prediction = 1 if prediction_proba >= 0.5 else 0 

            if prediction == 1:
                prob_percent = f"{prediction_proba * 100:.2f}%"
                return render_template('chance.html', probability=prob_percent)
            else:
                return render_template('noChance.html')

        except Exception as e:
            print(f"An unexpected error occurred during prediction: {e}")
            return render_template('noChance.html', message=f"Prediction Error: {e}. Check console for details.")

    return render_template('index.html')

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
