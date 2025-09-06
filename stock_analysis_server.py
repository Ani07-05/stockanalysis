from flask import Flask, request, send_file, jsonify
import os
import yfinance as yf
import pandas as pd
from finrobot.functional.quantitative import linear_regression_forecast
from finrobot.functional.stock_analysis import analyze_stock

app = Flask(__name__)

REPORTS_DIR = 'generated_reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({'error': 'Missing ticker'}), 400
    # Analyze stock (implement this function using your codebase)
    analysis = analyze_stock(ticker)
    # Read the actual content from the generated files
    output = {}
    for key, msg in analysis.items():
        # msg is like 'instruction & resources saved to /tmp/RELIANCE.NS_income.txt'
        if 'to ' in msg:
            path = msg.split('to ')[-1].strip()
            try:
                with open(path, 'r') as f:
                    output[key] = f.read()
            except Exception as e:
                output[key] = f"[Error reading {path}: {e}]"
        else:
            output[key] = msg
    return jsonify({"status": "success", "analysis": output})

# New endpoint for stock analysis and prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({'error': 'Missing ticker'}), 400

    try:
        # Download one year of historical data
        end_date = pd.to_datetime('today')
        start_date = end_date - pd.DateOffset(years=1)
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            return jsonify({'error': f'No data found for ticker {ticker}'}), 404

        # Get the prediction
        prediction_text = linear_regression_forecast(stock_data)
        
        return jsonify({"status": "success", "result": prediction_text})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
