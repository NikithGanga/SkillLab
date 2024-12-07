from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
from save_load_model import load_model
from data_collection import get_stock_data
from data_preprocessing import preprocess_data
from feature_engineering import add_features
from labeling import label_risk
from flask_mail import Mail, Message
import traceback
import logging

app = Flask(__name__)

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'tle.nikith123@example.com'  # Your email address
app.config['MAIL_PASSWORD'] = 'Nikith@123'          # Your email password
# app.config['MAIL_DEFAULT_SENDER'] = 'your_email@example.com'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form['ticker'].upper()
    try:
        # Get stock data and analyze
        data = get_stock_data(ticker)
        data = preprocess_data(data)
        data = add_features(data)
        data = label_risk(data)

        # Load the model
        model = load_model()

        # Get features for prediction
        features = ['Daily Return', 'Volatility', 'MA50', 'MA200']
        latest_data = data[features].iloc[-1]

        # Make prediction
        risk_level = model.predict(latest_data.values.reshape(1, -1))[0]

        # Format dates and prices for the chart (last 30 days)
        # Convert index to datetime if it's not already
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        dates = [d.strftime('%Y-%m-%d') for d in data.index[-30:]]
        prices = data['Close'].tail(30).tolist()

        # Send email if the risk level is low
        if risk_level.lower() == 'low':
            send_email(
                subject=f"Low Risk Alert for {ticker}",
                # Replace with actual recipient
                recipients=['nikithganga123@gmail.com'],
                body=f"Stock Analysis Alert:\n\n"
                     f"The stock {ticker} has been analyzed with a risk level of 'Low'.\n"
                     f"Details:\n"
                     f"Current Price: ${data['Close'].iloc[-1]:.2f}\n"
                     f"Volatility: {data['Volatility'].iloc[-1]*100:.2f}%\n"
                     f"Daily Return: {data['Daily Return'].iloc[-1]*100:.2f}%\n"
                     f"Stay informed and take action as needed."
            )

        return jsonify({
            'risk_level': risk_level,
            'current_price': f"{data['Close'].iloc[-1]:.2f}",
            'volatility': f"{data['Volatility'].iloc[-1]*100:.2f}",
            'daily_return': f"{data['Daily Return'].iloc[-1]*100:.2f}",
            'dates': dates,
            'prices': prices
        })
    except Exception as e:
        # Log the full error on the server
        logging.error(traceback.format_exc())
        return jsonify({'error': 'An internal error has occurred.'}), 400


def send_email(subject, recipients, body):
    """Send an email with Flask-Mail."""
    try:
        msg = Message(subject, recipients=recipients, body=body)
        mail.send(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


@app.route('/api/analyze/<ticker>', methods=['GET'])
def analyze_api(ticker):
    try:
        # Get stock data and analyze
        data = get_stock_data(ticker.upper())
        data = preprocess_data(data)
        data = add_features(data)
        data = label_risk(data)

        # Load the model
        model = load_model()

        # Get features for prediction
        features = ['Daily Return', 'Volatility', 'MA50', 'MA200']
        latest_data = data[features].iloc[-1]

        # Make prediction
        risk_level = model.predict(latest_data.values.reshape(1, -1))[0]

        # Prepare API response
        response = {
            'ticker': ticker.upper(),
            'analysis': {
                'risk_level': int(risk_level),
                'current_price': float(data['Close'].iloc[-1]),
                'volatility': float(data['Volatility'].iloc[-1]),
                'daily_return': float(data['Daily Return'].iloc[-1]),
                'last_updated': data.index[-1].isoformat()
            },
            'historical_data': {
                'dates': [d.isoformat() for d in data.index[-30:]],
                'prices': [float(p) for p in data['Close'].tail(30)]
            }
        }

        return jsonify(response)
    except Exception as e:
        # Log the full error on the server
        logging.error(traceback.format_exc())
        return jsonify({
            'error': 'An internal error has occurred.',
            'ticker': ticker.upper()
        }), 400


if __name__ == '__main__':
    app.run(debug=True)
