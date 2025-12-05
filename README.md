# House_Price_Prediction_Using_ML

![Uploading Screenshot (108).png…]()
🏠 House Price Estimator
A modern, responsive web application that predicts house prices using machine learning with a beautiful, user-friendly interface.

📊 Core Prediction
Machine Learning Model: Trained on housing data for accurate price predictions

Real-time Conversion: Automatic USD to INR conversion with live exchange rates

Price in Words: Converts numerical price to Indian numbering system words (e.g., "Two Crore Thirty Lakh")


Prerequisites
Python 3.8+

Flask

Scikit-learn

Pandas, NumPy

Installation
Clone the repository

bash
git clone https://github.com/yourusername/house-price-estimator.git
cd house-price-estimator
Create virtual environment

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Prepare model files

bash
# Place your trained model files in src/ directory
# - model.pkl (trained ML model)
# - preprocess.pkl (preprocessing metadata)
Run the application

bash
python app.py
Open in browser

text
http://localhost:5000
