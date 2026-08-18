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


# Columns the model actually needs as input (raw, pre-preprocessing).
# fraud_bool and device_fraud_count are deliberately excluded here:
# fraud_bool is the target (never required as input), device_fraud_count
# is dropped inside preprocess() because it's constant/uninformative.
REQUIRED_INPUT_COLUMNS = (
    config['continuous_cols']
    + config['categorical_cols_for_onehot']
    + ['email_is_free', 'phone_home_valid', 'phone_mobile_valid',
       'has_other_cards', 'foreign_request', 'keep_alive_session', 'month']
)


@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    """
    Accepts a CSV upload of one or more accounts (multipart/form-data, field name 'file').
    Each row = one account, with all raw feature columns except 'fraud_bool'.

    If the CSV happens to include 'fraud_bool' (e.g. a labeled test set), it is
    excluded from the model input and instead used to report accuracy metrics
    alongside the predictions -- it is never required for real/unlabeled uploads.

    Query param / form field threshold_mode: "high_recall" (default) or "high_precision".
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': "Please upload a CSV file using the 'file' field."}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No CSV file selected.'}), 400
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files are allowed.'}), 400

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Unable to read CSV file: {e}'}), 400

        if len(df) == 0:
            return jsonify({'success': False, 'error': 'The uploaded CSV has no rows.'}), 400

        missing_cols = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing_cols:
            return jsonify({
                'success': False,
                'error': 'Required feature columns are missing from the CSV.',
                'missing_columns': missing_cols
            }), 400

        threshold_mode = request.form.get('threshold_mode') or request.args.get('threshold_mode', 'high_recall')
        threshold_key = 'threshold_high_precision' if threshold_mode == 'high_precision' else 'threshold_high_recall'
        threshold = config[threshold_key]

        # Keep ground truth (if present) separately -- never fed to the model
        has_labels = 'fraud_bool' in df.columns
        y_true = df['fraud_bool'].copy() if has_labels else None

        model_input = df.drop(columns=['fraud_bool'], errors='ignore')
        processed = preprocess(model_input)
        proba = model.predict_proba(processed)[:, 1]
        prediction = np.where(proba >= threshold, 'Fraud', 'Not Fraud')

        accounts = []
        for i in range(len(df)):
            record = {
                'row_id': i,                      # CSV has no ID column -- reference by row position
                'fraud_probability': round(float(proba[i]), 4),
                'prediction': prediction[i],
            }
            if has_labels:
                record['actual'] = 'Fraud' if int(y_true.iloc[i]) == 1 else 'Not Fraud'
            accounts.append(record)

        response = {
            'success': True,
            'total_accounts': int(len(df)),
            'predicted_fraud': int((prediction == 'Fraud').sum()),
            'predicted_not_fraud': int((prediction == 'Not Fraud').sum()),
            'threshold_used': threshold,
            'threshold_mode': threshold_mode,
            'accounts': accounts,
        }

        # Bonus: if this was a labeled test set, report real accuracy metrics
        if has_labels:
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            y_pred = (prediction == 'Fraud').astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            response['evaluation'] = {
                'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                'f1_score': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                'confusion_matrix': {'true_negative': int(tn), 'false_positive': int(fp),
                                      'false_negative': int(fn), 'true_positive': int(tp)}
            }

        return jsonify(response)

    except KeyError as e:
        return jsonify({'success': False, 'error': f'Missing or unrecognized field: {e}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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
