# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import requests   # pip install requests

app = Flask(__name__)
ROOT = os.path.dirname(__file__)
MODEL = joblib.load(os.path.join(ROOT, '..', 'src', 'model.pkl'))
META = joblib.load(os.path.join(ROOT, '..', 'src', 'preprocess.pkl'))

numeric_features = META['numeric_features']
cat_features = META['cat_features']

# --- Settings ---
USE_LIVE_RATE = True           # Set False to always use FIXED_USD_TO_INR
FIXED_USD_TO_INR = 90.0        # fallback / manual rate (update as needed)
EXCHANGE_API = "https://api.exchangerate.host/latest?base=USD&symbols=INR"

def get_usd_to_inr_rate():
    """Return (rate, source) where source is 'live' or 'fixed'."""
    if not USE_LIVE_RATE:
        return FIXED_USD_TO_INR, "fixed"
    try:
        resp = requests.get(EXCHANGE_API, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        rate = data.get("rates", {}).get("INR")
        if rate is None:
            return FIXED_USD_TO_INR, "fixed"
        return float(rate), "live"
    except Exception:
        return FIXED_USD_TO_INR, "fixed"

def convert_to_indian_words(number):
    """Convert integer number to words in Indian numbering system."""
    if number == 0:
        return "Zero"
    units = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
             "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

    def two_digits(n):
        if n == 0:
            return ""
        if n < 10:
            return units[n]
        if 10 <= n < 20:
            return teens[n - 10]
        return (tens[n // 10] + (" " + units[n % 10] if n % 10 != 0 else "")).strip()

    def three_digits(n):
        if n == 0:
            return ""
        if n < 100:
            return two_digits(n)
        rem = n % 100
        return (units[n // 100] + " Hundred" + (" " + two_digits(rem) if rem else "")).strip()

    crore = number // 10000000
    number %= 10000000
    lakh = number // 100000
    number %= 100000
    thousand = number // 1000
    number %= 1000
    hundred = number  # 0-999

    parts = []
    if crore:
        parts.append((two_digits(crore) + " Crore").strip())
    if lakh:
        parts.append((two_digits(lakh) + " Lakh").strip())
    if thousand:
        parts.append((two_digits(thousand) + " Thousand").strip())
    if hundred:
        parts.append(three_digits(hundred).strip())

    return " ".join([p for p in parts if p]).strip()

@app.route('/')
def index():
    return render_template('index.html', prediction=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form values
    data = {}
    # numeric fields
    for f in numeric_features:
        val = request.form.get(f)
        try:
            data[f] = float(val) if val not in [None, ''] else np.nan
        except:
            data[f] = np.nan
    # categorical fields
    for f in cat_features:
        val = request.form.get(f)
        data[f] = val if val not in [None, ''] else np.nan

    df = pd.DataFrame([data], columns=(numeric_features + cat_features))

    # predict (model output assumed USD)
    try:
        usd_pred = float(MODEL.predict(df)[0])
    except Exception as e:
        # Return JSON for AJAX or render with error
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': f"Prediction error: {e}"})
        return render_template('index.html', 
                               prediction=None,
                               error=f"Prediction error: {e}",
                               form_data=request.form)

    # get conversion rate (numeric)
    rate, source = get_usd_to_inr_rate()
    inr_pred = usd_pred * rate

    # format INR nicely with 2 decimals
    inr_text = f"₹{inr_pred:,.2f}"

    # convert INR integer part to words
    inr_int = int(round(inr_pred))
    inr_words = convert_to_indian_words(inr_int)

    # Prepare result data
    result = {
        'price_inr': inr_text,
        'price_words': inr_words,
        'usd_price': f"${usd_pred:,.2f}",
        'exchange_rate': rate,
        'rate_source': source,
        'price_compact': f"₹{inr_pred:,.0f}"  # For compact display
    }
    
    # Return JSON for AJAX or render template
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(result)
    
    return render_template('index.html', 
                          prediction=result, 
                          error=None,
                          form_data=request.form)

if __name__ == '__main__':
    app.run(debug=True)