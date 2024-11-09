from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json  # To load feature columns

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('gradient_boosting_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load feature columns from JSON file
with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# Define columns that were scaled
scale_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        missing_cols = set(feature_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # Assign default value or handle appropriately
        
        # Reorder columns to match training
        input_df = input_df[feature_columns]
        
        # Feature Scaling
        input_df[scale_cols] = scaler.transform(input_df[scale_cols])
        
        # Handle any additional preprocessing (e.g., encoding)
        # If you performed One-Hot Encoding during training, ensure consistency here
        # For example, if 'Geography_Germany' and 'Geography_Spain' are expected:
        geography_cols = ['Geography_Germany', 'Geography_Spain']
        for col in geography_cols:
            if col not in input_df.columns:
                input_df[col] = 0  # Assign 0 if not present
        
        # Predict using the model
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]
        
        # Return prediction results
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(prediction_proba[0])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
