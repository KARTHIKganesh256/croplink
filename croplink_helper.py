import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "croplink_model.pkl"
ENCODERS_PATH = "croplink_encoders.pkl"
STATES_DISTRICTS_JSON = os.path.join("croplink", "states_districts.json")

def load_states_districts():
    if os.path.exists(STATES_DISTRICTS_JSON):
        with open(STATES_DISTRICTS_JSON, "r") as f:
            return json.load(f)
    else:
        # Fallback in case of missing file
        return {"Telangana": ["Hyderabad", "Rangareddy"], "Andhra Pradesh": ["Anantapur", "Kurnool"]}

def generate_agronomic_data(num_samples=3000):
    np.random.seed(42)
    distData = load_states_districts()
    
    all_states = sorted(list(distData.keys()))
    all_districts = []
    for dists in distData.values():
        all_districts.extend(dists)
    all_districts = sorted(list(set(all_districts)))
    
    seasons = ['Kharif', 'Rabi', 'Summer']
    crops = ['Rice', 'Wheat', 'Maize']
    
    # Random uniform values for features
    state_choices = np.random.choice(all_states, num_samples)
    
    # Make sure district chosen is valid for selected state
    district_choices = []
    for s in state_choices:
        dists = distData[s]
        district_choices.append(np.random.choice(dists))
        
    year_choices = np.random.randint(2015, 2026, num_samples)
    season_choices = np.random.choice(seasons, num_samples)
    crop_choices = np.random.choice(crops, num_samples)
    
    area_choices = np.random.uniform(1.0, 10.0, num_samples) # hectares
    soil_moisture_choices = np.random.uniform(20.0, 60.0, num_samples) # %
    temp_choices = np.random.uniform(15.0, 38.0, num_samples) # °C
    humidity_choices = np.random.uniform(40.0, 80.0, num_samples) # %
    ph_choices = np.random.uniform(4.5, 8.5, num_samples)
    rainfall_choices = np.random.uniform(100.0, 1200.0, num_samples) # mm
    
    n_choices = np.random.uniform(30.0, 180.0, num_samples) # kg/ha
    p_choices = np.random.uniform(10.0, 80.0, num_samples) # kg/ha
    k_choices = np.random.uniform(30.0, 220.0, num_samples) # kg/ha
    
    pest_choices = np.random.randint(0, 5, num_samples) # 0-4
    disease_choices = np.random.randint(0, 5, num_samples) # 0-4
    
    # Calculate yield based on logical crop rules
    yields = []
    for i in range(num_samples):
        crop = crop_choices[i]
        
        # Base yields
        if crop == 'Rice':
            base = 3500.0
            opt_temp = 28.0
            opt_ph = 6.2
            opt_moisture = 48.0
            opt_rain = 800.0
        elif crop == 'Wheat':
            base = 3200.0
            opt_temp = 20.0
            opt_ph = 6.8
            opt_moisture = 38.0
            opt_rain = 400.0
        else: # Maize
            base = 3000.0
            opt_temp = 25.0
            opt_ph = 6.5
            opt_moisture = 42.0
            opt_rain = 600.0
            
        # Temperature impact (bell curve)
        temp_factor = np.exp(-0.015 * (temp_choices[i] - opt_temp)**2)
        
        # Soil pH impact
        ph_factor = np.exp(-0.15 * (ph_choices[i] - opt_ph)**2)
        
        # Soil moisture impact
        moisture_factor = np.exp(-0.01 * (soil_moisture_choices[i] - opt_moisture)**2)
        
        # Rainfall impact
        rain_factor = np.exp(-0.0005 * (rainfall_choices[i] - opt_rain)**2)
        
        # NPK Fertilizer health factor
        # Balanced target: N=120, P=50, K=90
        n_factor = 1.0 - 0.3 * (abs(n_choices[i] - 120.0) / 120.0)
        p_factor = 1.0 - 0.3 * (abs(p_choices[i] - 50.0) / 50.0)
        k_factor = 1.0 - 0.3 * (abs(k_choices[i] - 90.0) / 90.0)
        npk_factor = (n_factor + p_factor + k_factor) / 3.0
        
        # Pest and disease reduction
        pest_penalty = 1.0 - (pest_choices[i] * 0.08)
        disease_penalty = 1.0 - (disease_choices[i] * 0.08)
        
        # Calculate yield
        predicted_yield = base * temp_factor * ph_factor * moisture_factor * rain_factor * npk_factor * pest_penalty * disease_penalty
        
        # Add random noise (+/- 5%)
        noise = 1.0 + np.random.uniform(-0.05, 0.05)
        predicted_yield *= noise
        
        # Clamp to realistic crop boundaries
        predicted_yield = max(500.0, min(predicted_yield, 6500.0))
        yields.append(predicted_yield)
        
    df = pd.DataFrame({
        "state": state_choices,
        "district": district_choices,
        "crop_year": year_choices,
        "season": season_choices,
        "crop": crop_choices,
        "area": area_choices,
        "soil_moisture": soil_moisture_choices,
        "temperature": temp_choices,
        "humidity": humidity_choices,
        "ph": ph_choices,
        "rainfall": rainfall_choices,
        "nutrient_n": n_choices,
        "nutrient_p": p_choices,
        "nutrient_k": k_choices,
        "pest_level": pest_choices,
        "disease_level": disease_choices,
        "yield": yields
    })
    
    return df

def get_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODERS_PATH)
            return model, encoders
        except Exception:
            pass # Re-train if load fails
            
    print("Training detailed Indian Crop model...")
    df = generate_agronomic_data()
    
    encoders = {}
    categorical_cols = ["state", "district", "season", "crop"]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop(columns=["yield"])
    y = df["yield"]
    
    model = RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42)
    model.fit(X, y)
    
    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoders, ENCODERS_PATH)
        print("Model trained and encoders saved successfully.")
    except Exception as write_err:
        print(f"Warning: Could not save model/encoders to disk (read-only filesystem): {write_err}")
    
    return model, encoders

if __name__ == "__main__":
    get_or_train_model()
