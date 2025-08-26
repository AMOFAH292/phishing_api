import re
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('PHISHING_MODEL/model.pkl')
    scaler = joblib.load('PHISHING_MODEL/scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    raise

def extract_features(url):
    features = {
        'url_length': len(url),
        'has_ip': int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', url))),
        'has_at_symbol': int('@' in url),
        'has_hyphen': int('-' in url),
        'num_dots': url.count('.'),
        'has_https': int('https' in url),
        'count_www': url.count('www'),
        'count_slash': url.count('/'),
        'count_percent': url.count('%'),
    }
    return list(features.values())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Missing "url" in request body'}), 400
        
        url = data['url']
        
        # Validate URL is a non-empty string
        if not isinstance(url, str) or not url.strip():
            return jsonify({'error': 'Invalid or empty URL'}), 400
        
        # Extract features
        features = extract_features(url)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)
        
        # Prepare response
        result = {
            'url': url,
            'prediction': 'phishing' if prediction[0] == 1 else 'legit',
            'confidence': f"{prediction_proba[0][1]*100:.2f}"
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)