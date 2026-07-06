import os
import json
import logging
import numpy as np
import pandas as pd
import requests
import joblib
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
from geopy.geocoders import Nominatim
import croplink_helper

try:
    from twilio.rest import Client
    twilio_available = True
except ImportError:
    twilio_available = False

# Configure Logging
logging.basicConfig(
    filename='app_unified.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Flask
# We serve static files directly from the root workspace folder to keep image paths like "./correlation_heatmap.png" working
app = Flask(__name__, static_folder='.', static_url_path='/static')

# Twilio Client (Safe configuration via environment variables)
TWILIO_SID = os.environ.get('TWILIO_SID', '')
TWILIO_TOKEN = os.environ.get('TWILIO_TOKEN', '')
twilio_client = None
if twilio_available:
    try:
        twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
        logging.info("Twilio client initialized.")
    except Exception as e:
        logging.warning(f"Twilio initialization failed: {e}. SMS features will be bypassed.")
else:
    logging.warning("Twilio package not installed. SMS features will be bypassed.")


# Load models and encoders
GLOBAL_MODEL_PATH = 'best_model.pkl'
ENCODER_MAPPINGS_PATH = os.path.join('croplink', 'encoder_mappings.json')

global_model = None
global_area_encoder = None
global_item_encoder = None

indian_model = None
indian_encoders = None

# Load global model
try:
    if os.path.exists(GLOBAL_MODEL_PATH):
        global_model = joblib.load(GLOBAL_MODEL_PATH)
        logging.info("Global crop yield prediction model loaded.")
    else:
        logging.error("best_model.pkl not found! Yield prediction will fail.")
except Exception as e:
    logging.error(f"Error loading best_model.pkl: {e}")

# Reconstruct global encoders from JSON
try:
    if os.path.exists(ENCODER_MAPPINGS_PATH):
        with open(ENCODER_MAPPINGS_PATH, 'r') as f:
            mappings = json.load(f)
        
        # Area encoder
        global_area_encoder = LabelEncoder()
        global_area_encoder.classes_ = np.array(list(mappings['Area'].keys()))
        
        # Item encoder
        global_item_encoder = LabelEncoder()
        global_item_encoder.classes_ = np.array(list(mappings['Item'].keys()))
        logging.info("Global Area and Item label encoders reconstructed from mappings.")
    else:
        logging.warning("Encoder mappings not found. Creating encoders dynamically from dataset...")
        if os.path.exists('yield_df.csv'):
            df_temp = pd.read_csv('yield_df.csv')
            global_area_encoder = LabelEncoder().fit(df_temp['Area'])
            global_item_encoder = LabelEncoder().fit(df_temp['Item'])
            logging.info("Global encoders created dynamically.")
except Exception as e:
    logging.error(f"Failed to load global encoders: {e}")

# Load Indian model and encoders
try:
    indian_model, indian_encoders = croplink_helper.get_or_train_model()
    logging.info("Indian crop model and encoders loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Indian crop model: {e}")

# --- REST APIs ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/states', methods=['GET'])
def get_states():
    try:
        states_districts = croplink_helper.load_states_districts()
        return jsonify({"states": sorted(list(states_districts.keys()))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/districts/<state>', methods=['GET'])
def get_districts(state):
    try:
        states_districts = croplink_helper.load_states_districts()
        if state in states_districts:
            return jsonify({"state": state, "districts": sorted(states_districts[state])})
        else:
            return jsonify({"error": "State not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-coordinates', methods=['POST'])
def get_coordinates():
    try:
        data = request.get_json()
        if not data or 'location' not in data:
            return jsonify({"error": "Missing location string"}), 400
        
        geolocator = Nominatim(user_agent="croplink_ai_platform")
        location = geolocator.geocode(data['location'], timeout=10)
        if not location:
            return jsonify({"error": "Location not found"}), 404
            
        return jsonify({
            "latitude": location.latitude,
            "longitude": location.longitude,
            "display_name": location.address
        })
    except Exception as e:
        logging.error(f"Geocoding error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/fetch-weather', methods=['POST'])
def fetch_weather():
    try:
        data = request.get_json()
        if not data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({"error": "Missing latitude or longitude"}), 400
            
        lat = data['latitude']
        lon = data['longitude']
        api_key = data.get('api_key') or "8d757d5970c0c7a4d5386db1a03f443b" # Default fallback key
        
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                weather_data = resp.json()
                temp = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                
                # Annual rain proxy logic from OpenWeatherMap (rough approximation)
                rain_mm = 600.0 # Default fallback
                if 'rain' in weather_data:
                    if '1h' in weather_data['rain']:
                        rain_mm = weather_data['rain']['1h'] * 8760 * 0.1 # scaled
                    elif '3h' in weather_data['rain']:
                        rain_mm = weather_data['rain']['3h'] * 2920 * 0.1 # scaled
                
                return jsonify({
                    "temperature_celsius": temp,
                    "humidity_percent": humidity,
                    "rainfall_mm": rain_mm
                })
        except Exception as api_err:
            logging.warning(f"Live weather API error: {api_err}. Serving fallback mock data.")
            
        # Mock weather fallback if API key is invalid or request fails
        # Determine based on latitude (e.g. tropical vs temperate)
        is_tropical = abs(lat) < 23.5
        mock_temp = np.random.uniform(24.0, 32.0) if is_tropical else np.random.uniform(14.0, 22.0)
        mock_humidity = np.random.uniform(50, 85) if is_tropical else np.random.uniform(40, 70)
        mock_rain = np.random.uniform(500, 1500) if is_tropical else np.random.uniform(200, 800)
        
        return jsonify({
            "temperature_celsius": round(mock_temp, 1),
            "humidity_percent": int(mock_humidity),
            "rainfall_mm": round(mock_rain, 1),
            "is_mock": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fetch-soil', methods=['POST'])
def fetch_soil():
    try:
        data = request.get_json()
        if not data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({"error": "Missing latitude or longitude"}), 400
            
        lat = data['latitude']
        lon = data['longitude']
        
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=phh2o&property=clay&property=nitrogen&depth=0-5cm&value=mean"
        
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soil_data = resp.json()
                layers = soil_data.get('properties', {}).get('layers', [])
                ph = None
                clay = None
                n_val = None
                
                for layer in layers:
                    if layer['name'] == 'phh2o':
                        ph = layer['depths'][0]['values']['mean'] / 10.0
                    elif layer['name'] == 'clay':
                        clay = layer['depths'][0]['values']['mean'] / 10.0
                    elif layer['name'] == 'nitrogen':
                        n_val = layer['depths'][0]['values']['mean'] / 100.0
                
                # Determine Soil Type
                soil_type = 'Loam Soil'
                if clay is not None:
                    if clay > 35:
                        soil_type = 'Clay/Black Soil'
                    elif clay > 20:
                        soil_type = 'Alluvial Soil'
                    elif clay > 10:
                        soil_type = 'Red Soil'
                    else:
                        soil_type = 'Sandy Soil'
                
                nitrogen_kg_ha = (n_val * 10) if n_val else 80.0
                
                return jsonify({
                    "ph": ph or 6.5,
                    "clay_percent": clay or 22.0,
                    "soil_type": soil_type,
                    "nitrogen_kg_per_ha": nitrogen_kg_ha
                })
        except Exception as api_err:
            logging.warning(f"Soil grids API query error: {api_err}. Serving fallback mock data.")
            
        # Mock soil fallback in case ISRIC is down or location is out of bounds
        mock_ph = round(np.random.uniform(5.5, 7.5), 1)
        mock_clay = round(np.random.uniform(12.0, 38.0), 1)
        
        soil_types = ['Black Soil', 'Red Soil', 'Alluvial Soil', 'Sandy Soil']
        mock_soil = np.random.choice(soil_types)
        mock_n = round(np.random.uniform(60, 140), 1)
        
        return jsonify({
            "ph": mock_ph,
            "clay_percent": mock_clay,
            "soil_type": mock_soil,
            "nitrogen_kg_per_ha": mock_n,
            "is_mock": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/global', methods=['POST'])
def predict_global():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        required_fields = ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400
            
        area = data['Area']
        item = data['Item']
        
        # Verify values in encoders
        if area not in global_area_encoder.classes_:
            return jsonify({"error": f"Country/Area '{area}' is not trained in this model."}), 400
        if item not in global_item_encoder.classes_:
            return jsonify({"error": f"Crop '{item}' is not trained in this model."}), 400
            
        area_enc = int(global_area_encoder.transform([area])[0])
        item_enc = int(global_item_encoder.transform([item])[0])
        
        features = pd.DataFrame([[
            area_enc,
            item_enc,
            int(data['Year']),
            float(data['average_rain_fall_mm_per_year']),
            float(data['pesticides_tonnes']),
            float(data['avg_temp'])
        ]], columns=['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'])
        
        # Predict
        if global_model is not None:
            prediction = float(global_model.predict(features)[0])
        else:
            # Fallback to linear regression coefficients from model_coefficients.json (164 bytes)
            logging.warning("global_model is not loaded. Falling back to Linear Regression coefficients.")
            coefs_file = 'model_coefficients.json'
            if not os.path.exists(coefs_file):
                coefs_file = os.path.join('croplink', 'model_coefficients.json')
                
            if os.path.exists(coefs_file):
                with open(coefs_file, 'r') as f:
                    coef_data = json.load(f)
                coef = coef_data['coef']
                intercept = coef_data['intercept']
                
                # Linear Regression Equation:
                # intercept + coef[0]*Area + coef[1]*Item + coef[2]*Year + coef[3]*Rain + coef[4]*Pesticides + coef[5]*Temp
                pred_val = (
                    intercept +
                    coef[0] * area_enc +
                    coef[1] * item_enc +
                    coef[2] * int(data['Year']) +
                    coef[3] * float(data['average_rain_fall_mm_per_year']) +
                    coef[4] * float(data['pesticides_tonnes']) +
                    coef[5] * float(data['avg_temp'])
                )
                prediction = float(pred_val)
            else:
                raise FileNotFoundError("Neither best_model.pkl nor model_coefficients.json were found.")
        
        # Send SMS via Twilio (fail-safe)
        if twilio_client:
            try:
                msg_body = f"CropLink Update: Global model yield prediction for {item} in {area} is {prediction:.2f} hg/ha."
                message = twilio_client.messages.create(
                    body=msg_body,
                    from_='+18777804236',
                    to='+918106415496'
                )
                logging.info(f"Twilio SMS sent: {message.sid}")
            except Exception as sms_err:
                logging.warning(f"SMS sending failed: {sms_err}")
                
        return jsonify({
            "predicted_yield_hg_per_ha": round(prediction, 2),
            "predicted_yield_kg_per_ha": round(prediction * 0.1, 2) # hg/ha to kg/ha conversion (1 hg = 0.1 kg)
        })
    except Exception as e:
        logging.error(f"Global prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/indian', methods=['POST'])
def predict_indian():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        # Extract inputs
        state = data.get('state')
        district = data.get('district')
        crop_year = int(data.get('crop_year', 2025))
        season = data.get('season')
        crop = data.get('crop')
        area = float(data.get('area', 1.0))
        soil_moisture = float(data.get('soil_moisture', 40.0))
        temperature = float(data.get('temperature', 25.0))
        humidity = float(data.get('humidity', 60.0))
        ph = float(data.get('ph', 6.5))
        rainfall = float(data.get('rainfall', 600.0))
        nutrient_n = float(data.get('nutrient_n', 100.0))
        nutrient_p = float(data.get('nutrient_p', 40.0))
        nutrient_k = float(data.get('nutrient_k', 80.0))
        pest_level = int(data.get('pest_level', 1))
        disease_level = int(data.get('disease_level', 1))
        
        # Apply encoders
        try:
            state_enc = int(indian_encoders['state'].transform([state])[0])
            district_enc = int(indian_encoders['district'].transform([district])[0])
            season_enc = int(indian_encoders['season'].transform([season])[0])
            crop_enc = int(indian_encoders['crop'].transform([crop])[0])
        except Exception as enc_err:
            return jsonify({"error": f"Invalid categories for encoding: {enc_err}"}), 400
            
        # Features array
        features = np.array([[
            state_enc,
            district_enc,
            crop_year,
            season_enc,
            crop_enc,
            area,
            soil_moisture,
            temperature,
            humidity,
            ph,
            rainfall,
            nutrient_n,
            nutrient_p,
            nutrient_k,
            pest_level,
            disease_level
        ]])
        
        # Predict using estimators for uncertainty metrics
        tree_preds = np.array([tree.predict(features)[0] for tree in indian_model.estimators_])
        pred_yield = float(np.mean(tree_preds))
        pred_production = float(pred_yield * area)
        uncertainty = float(np.std(tree_preds) / pred_yield * 100) if pred_yield != 0 else 0.0
        
        return jsonify({
            "predicted_yield_kg_per_ha": round(pred_yield, 2),
            "total_production_kg": round(pred_production, 2),
            "prediction_uncertainty_percent": round(uncertainty, 2)
        })
    except Exception as e:
        logging.error(f"Indian detailed prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    try:
        # Load yield df stats
        stats = {
            "total_records": 28242,
            "average_yield": 77053, # hg/ha
            "min_temp": 1.30,
            "max_temp": 30.65,
            "min_rainfall": 51.0,
            "max_rainfall": 3240.0
        }
        
        # Try loading actual dataset details dynamically
        if os.path.exists('yield_df.csv'):
            try:
                df = pd.read_csv('yield_df.csv')
                stats["total_records"] = int(len(df))
                stats["average_yield"] = float(df['hg/ha_yield'].mean())
                stats["min_temp"] = float(df['avg_temp'].min())
                stats["max_temp"] = float(df['avg_temp'].max())
                stats["min_rainfall"] = float(df['average_rain_fall_mm_per_year'].min())
                stats["max_rainfall"] = float(df['average_rain_fall_mm_per_year'].max())
            except Exception as e:
                logging.error(f"Failed parsing yield_df.csv for stats: {e}")
                
        # Load encoder options for global predictor selects
        global_areas = sorted(list(global_area_encoder.classes_)) if global_area_encoder else []
        global_crops = sorted(list(global_item_encoder.classes_)) if global_item_encoder else []
        
        # Load model metrics from CSV
        model_metrics = []
        if os.path.exists('model_results.csv'):
            try:
                df_metrics = pd.read_csv('model_results.csv')
                model_metrics = df_metrics.to_dict(orient='records')
            except Exception as e:
                logging.error(f"Failed parsing model_results.csv: {e}")
                
        return jsonify({
            "stats": stats,
            "areas": global_areas,
            "crops": global_crops,
            "model_metrics": model_metrics
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)