import pandas as pd
import pickle
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import json
import os

app = Flask(__name__)

# Load the KMeans model
try:
    with open('kmeans_superstore_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
except FileNotFoundError:
    print("Error: kmeans_superstore_model.pkl not found.")

# Initialize scaler with dataset ranges
scaler = StandardScaler()
training_data = pd.DataFrame({
    'Sales': [0.0, 22638.0],    # Replace with your min/max from df.describe()
    'Quantity': [1.0, 14.0],
    'Discount': [0.0, 0.8],
    'Profit': [-6599.0, 8399.0]
})
scaler.fit(training_data)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    sales = float(request.form['sales'])
    quantity = float(request.form['quantity'])
    discount = float(request.form['discount'])
    profit = float(request.form['profit'])

    input_data = np.array([[sales, quantity, discount, profit]])
    scaled_input = scaler.transform(input_data)
    cluster = kmeans.predict(scaled_input)[0]
    advice = get_cluster_advice(cluster)

    return render_template('result.html', cluster=cluster, advice=advice,
                          sales=sales, quantity=quantity, discount=discount, profit=profit)

@app.route('/visualize')
def visualize():
    # Hardcoded visualization data (replace with real stats if available)
    cluster_counts = [2500, 3000, 2000, 2494]
    total = sum(cluster_counts)
    pie_data = [round((count / total) * 100, 2) for count in cluster_counts]
    bar_data = {
        'Sales': [100, 500, 300, 700],
        'Quantity': [3, 5, 4, 6],
        'Discount': [0.1, 0.3, 0.2, 0.4],
        'Profit': [20, 50, -10, 80]
    }
    hist_data = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    return render_template('visualize.html', 
                          pie_data=json.dumps(pie_data),
                          bar_data=json.dumps(bar_data),
                          hist_data=json.dumps(hist_data))

def get_cluster_advice(cluster):
    advice_dict = {
        0: "Low sales and profit: Increase marketing and explore new customer segments.",
        1: "High sales, moderate profit: Optimize discount strategies to improve margins.",
        2: "High discounts, low profit: Reduce discounts and focus on cost efficiency.",
        3: "High sales and profit: Maintain strategy and explore upselling opportunities."
    }
    return advice_dict.get(cluster, "Review pricing and marketing strategies.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)