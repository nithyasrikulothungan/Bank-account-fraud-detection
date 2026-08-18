from flask import Flask, request, jsonify
from flask_cors import CORS
from catboost import CatBoostClassifier
import joblib
import json
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ---- CORS: allow the frontend (hosted on a different domain) to call this API ----
# For production, replace "*" with your actual frontend domain, e.g.:
# CORS(app, origins=["https://your-frontend-domain.com"])
CORS(app)

# ---- Load model + preprocessing artifacts once, at startup ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = CatBoostClassifier()
model.load_model(os.path.join(BASE_DIR, 'fraud_model.cbm'))

scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))

with open(os.path.join(BASE_DIR, 'feature_columns.json')) as f:
    feature_columns = json.load(f)

with open(os.path.join(BASE_DIR, 'model_config.json')) as f:
    config = json.load(f)

# Fail loudly at startup, not silently at request time, if config is incomplete
_required_config_keys = [
    'continuous_cols', 'categorical_cols_for_onehot', 'missing_sentinel_cols',
    'missing_value_medians', 'threshold_high_precision', 'threshold_high_recall'
]
_missing_keys = [k for k in _required_config_keys if k not in config]
if _missing_keys:
    raise RuntimeError(
        f"model_config.json is missing required keys: {_missing_keys}. "
        f"Regenerate it from the training notebook before deploying."
    )


def preprocess(df):
    """Apply the exact same preprocessing used during training."""
    df = df.copy()

    # Drop constant column if present
    if 'device_fraud_count' in df.columns:
        df = df.drop(columns=['device_fraud_count'])

    # Handle -1 sentinel missing values using SAVED training medians (not recomputed)
    for c in config['missing_sentinel_cols']:
        if c in df.columns:
            df[c + '_missing'] = (df[c] == -1).astype(int)
            df[c] = df[c].replace(-1, config['missing_value_medians'][c])
        else:
            # Column wasn't sent at all — treat as missing
            df[c + '_missing'] = 1
            df[c] = config['missing_value_medians'][c]

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=config['categorical_cols_for_onehot'])

    # Align to the exact training column set/order (missing dummy cols filled with 0)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Standardize continuous columns using the SAVED fitted scaler
    df[config['continuous_cols']] = scaler.transform(df[config['continuous_cols']])

    return df


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Fraud detection API is running'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON body: either a single record (dict) or a list of records.
    Example single record:
    {
        "income": 0.7, "name_email_similarity": 0.6, "prev_address_months_count": -1,
        "current_address_months_count": 50, "customer_age": 30, "days_since_request": 0.5,
        "intended_balcon_amount": 10.0, "payment_type": "AB", "zip_count_4w": 1200,
        "velocity_6h": 3000, "velocity_24h": 4000, "velocity_4w": 5000,
        "bank_branch_count_8w": 5, "date_of_birth_distinct_emails_4w": 3,
        "employment_status": "CA", "credit_risk_score": 130, "email_is_free": 1,
        "housing_status": "BC", "phone_home_valid": 1, "phone_mobile_valid": 1,
        "bank_months_count": 20, "has_other_cards": 1, "proposed_credit_limit": 500,
        "foreign_request": 0, "source": "INTERNET", "session_length_in_minutes": 5,
        "device_os": "windows", "keep_alive_session": 1,
        "device_distinct_emails_8w": 1, "month": 3
    }
    threshold_mode (optional, in query string or JSON): "high_recall" (default,
    recommended per the model report) or "high_precision"
    """
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({'error': 'Request body must be valid JSON.'}), 400

        # Default changed to high_recall (Broad Safety-Net Mode) to match the
        # deployment recommendation in the model report — override explicitly
        # with threshold_mode="high_precision" if the caller wants the stricter mode.
        threshold_mode = request.args.get('threshold_mode', 'high_recall')
        if isinstance(payload, dict) and 'threshold_mode' in payload:
            threshold_mode = payload.pop('threshold_mode')

        records = payload if isinstance(payload, list) else [payload]
        if len(records) == 0:
            return jsonify({'error': 'No records provided.'}), 400

        df = pd.DataFrame(records)

        processed = preprocess(df)
        proba = model.predict_proba(processed)[:, 1]

        threshold_key = 'threshold_high_precision' if threshold_mode == 'high_precision' else 'threshold_high_recall'
        threshold = config[threshold_key]

        prediction = np.where(proba >= threshold, 'Fraud', 'Not Fraud')

        results = []
        for i in range(len(df)):
            results.append({
                'fraud_probability': round(float(proba[i]), 4),
                'prediction': prediction[i],
                'threshold_used': threshold,
                'threshold_mode': threshold_mode
            })

        return jsonify({'results': results})

    except KeyError as e:
        return jsonify({'error': f'Missing or unrecognized field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
