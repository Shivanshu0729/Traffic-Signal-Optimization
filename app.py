import numpy as np
import json
import datetime
from flask import Flask, render_template, jsonify, request
from ga_traffic import optimize_signal_time
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

app = Flask(__name__)

# Load or create traffic prediction model
try:
    traffic_model = joblib.load('traffic_model.pkl')
except:
    traffic_model = LinearRegression()

# Function to simulate real-time traffic data with enhanced realism
def get_traffic_data():
    now = datetime.datetime.now()
    hour = now.hour
    minute = now.minute
    
    # Base values with time-based variation
    rush_hour_factor = 1 + 0.5 * np.sin((hour - 8) * np.pi / 8) if 7 <= hour <= 10 or 16 <= hour <= 19 else 1
    speed_variation = max(20, 70 - (30 * rush_hour_factor))
    
    return {
        'vehicle_speed': int(np.random.normal(speed_variation, 10)),  # More realistic speed distribution
        'traffic_density': int(np.random.poisson(30 * rush_hour_factor)),  # Poisson distribution for vehicle counts
        'further_road_traffic': int(np.random.random() < (0.3 * rush_hour_factor)),  # Rush hour affects probability
        'emergency_vehicle': int(np.random.random() < 0.03),  # 3% chance of emergency vehicle
        'pedestrian_crossing': int(np.random.random() < (0.1 + 0.2 * (hour in [8,12,17]))),  # Higher during peak hours
        'timestamp': str(now),
        'intersection_id': 'A1'  # Support for multiple intersections
    }

# Enhanced congestion analysis
def analyze_congestion(traffic_density):
    if traffic_density < 15:
        return "Very Low"
    elif traffic_density < 30:
        return "Low"
    elif traffic_density < 60:
        return "Moderate"
    elif traffic_density < 100:
        return "High"
    else:
        return "Severe"

# Predictive traffic modeling
def predict_traffic(traffic_data):
    # Convert timestamp to features
    now = datetime.datetime.now()
    features = np.array([
        now.hour,
        now.minute,
        now.weekday(),
        traffic_data['traffic_density'],
        traffic_data['vehicle_speed']
    ]).reshape(1, -1)
    
    # Predict next 15 minutes (simple linear regression for demo)
    try:
        prediction = traffic_model.predict(features)[0]
    except:
        prediction = traffic_data['traffic_density'] * 1.1  # Fallback
        
    return max(0, min(150, prediction))

# Enhanced signal optimization with multi-condition handling
def optimize_traffic(live_data):
    # Base optimization
    signal_time = optimize_signal_time(
        live_data['traffic_density'], 
        live_data['further_road_traffic']
    )
    
    # Emergency vehicle priority
    if live_data['emergency_vehicle']:
        signal_time = min(10, signal_time)  # Quick transition for emergency
        
    # Pedestrian accommodation
    if live_data['pedestrian_crossing']:
        signal_time = max(20, signal_time + 5)  # Extra time for pedestrians
        
    # Predictive adjustment
    predicted_density = predict_traffic(live_data)
    if predicted_density > live_data['traffic_density'] * 1.2:
        signal_time = min(120, signal_time * 1.2)
        
    return int(signal_time)

# Enhanced data logging with compression
def log_traffic_data(data):
    log_entry = {
        "timestamp": data['timestamp'],
        "vehicle_speed": data['vehicle_speed'],
        "traffic_density": data['traffic_density'],
        "congestion_level": analyze_congestion(data['traffic_density']),
        "optimized_signal_time": data['optimized_signal_time'],
        "emergency_vehicle": data['emergency_vehicle'],
        "pedestrian_crossing": data['pedestrian_crossing'],
        "intersection_id": data.get('intersection_id', 'A1')
    }
    
    try:
        with open("traffic_log.json", "r") as file:
            logs = json.load(file)
            # Keep only last 1000 entries to prevent large files
            logs = logs[-999:]
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []
    
    logs.append(log_entry)
    
    with open("traffic_log.json", "w") as file:
        json.dump(logs, file, indent=2)
        
    # Update prediction model periodically
    if len(logs) % 50 == 0:
        update_traffic_model(logs)

def update_traffic_model(logs):
    try:
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['weekday'] = df['timestamp'].dt.weekday
        
        X = df[['hour', 'minute', 'weekday', 'traffic_density']]
        y = df['vehicle_speed']
        
        global traffic_model
        traffic_model = LinearRegression().fit(X, y)
        joblib.dump(traffic_model, 'traffic_model.pkl')
    except Exception as e:
        print(f"Model update failed: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/traffic-analysis', methods=['GET'])
def traffic_analysis():
    live_data = get_traffic_data()
    live_data['optimized_signal_time'] = optimize_traffic(live_data)
    live_data['congestion_level'] = analyze_congestion(live_data['traffic_density'])
    live_data['predicted_density'] = predict_traffic(live_data)
    
    log_traffic_data(live_data)
    
    return jsonify(live_data)

@app.route('/historical-data', methods=['GET'])
def historical_data():
    try:
        with open("traffic_log.json", "r") as file:
            logs = json.load(file)
            return jsonify(logs[-100:])  # Return last 100 entries
    except:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')