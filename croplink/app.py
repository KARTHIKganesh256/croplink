from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import requests
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import uvicorn
import json

app = FastAPI(title="AgriPredict AI Backend")

# Load Indian states and districts data
districts_data = {
    "Andhra Pradesh": ["Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna", "Kurnool", "Prakasam", "Srikakulam", "Visakhapatnam", "Vizianagaram", "West Godavari", "YSR Kadapa"],
    "Arunachal Pradesh": ["Tawang", "West Kameng", "East Kameng", "Papum Pare", "Kurung Kumey", "Kra Daadi", "Lower Subansiri", "Upper Subansiri", "West Siang", "East Siang", "Siang", "Upper Siang", "Lower Siang", "Lower Dibang Valley", "Dibang Valley", "Anjaw", "Lohit", "Namsai", "Changlang", "Tirap", "Longding"],
    "Assam": ["Baksa", "Barpeta", "Biswanath", "Bongaigaon", "Cachar", "Charaideo", "Chirang", "Darrang", "Dhemaji", "Dhubri", "Dibrugarh", "Dima Hasao", "Goalpara", "Golaghat", "Hailakandi", "Jorhat", "Kamrup Metropolitan", "Kamrup", "Karbi Anglong", "Karimganj", "Kokrajhar", "Lakhimpur", "Majuli", "Morigaon", "Nagaon", "Nalbari", "Sivasagar", "Sonitpur", "South Salmara-Mankachar", "Tinsukia", "Udalguri", "West Karbi Anglong"],
    "Bihar": ["Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur", "Bhojpur", "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj", "Jamui", "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj", "Lakhisarai", "Madhepura", "Madhubani", "Munger", "Muzaffarpur", "Nalanda", "Nawada", "Patna", "Purnia", "Rohtas", "Saharsa", "Samastipur", "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan", "Supaul", "Vaishali", "West Champaran"],
    "Chhattisgarh": ["Balod", "Baloda Bazar", "Balrampur", "Bastar", "Bemetara", "Bijapur", "Bilaspur", "Dantewada", "Dhamtari", "Durg", "Gariaband", "Janjgir-Champa", "Jashpur", "Kabirdham", "Kanker", "Kondagaon", "Korba", "Korea", "Mahasamund", "Mungeli", "Narayanpur", "Raigarh", "Raipur", "Rajnandgaon", "Sukma", "Surajpur", "Surguja"],
    "Goa": ["North Goa", "South Goa"],
    "Gujarat": ["Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha", "Bharuch", "Bhavnagar", "Botad", "Chhota Udaipur", "Dahod", "Dang", "Devbhoomi Dwarka", "Gandhinagar", "Gir Somnath", "Jamnagar", "Junagadh", "Kheda", "Kutch", "Mahisagar", "Mehsana", "Morbi", "Narmada", "Navsari", "Panchmahal", "Patan", "Porbandar", "Rajkot", "Sabarkantha", "Surat", "Surendranagar", "Tapi", "Vadodara", "Valsad"],
    "Haryana": ["Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad", "Gurgaon", "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal", "Kurukshetra", "Mahendragarh", "Nuh", "Palwal", "Panchkula", "Panipat", "Rewari", "Rohtak", "Sirsa", "Sonipat", "Yamunanagar"],
    "Himachal Pradesh": ["Bilaspur", "Chamba", "Hamirpur", "Kangra", "Kinnaur", "Kullu", "Lahaul and Spiti", "Mandi", "Shimla", "Sirmaur", "Solan", "Una"],
    "Jharkhand": ["Bokaro", "Chatra", "Deoghar", "Dhanbad", "Dumka", "East Singhbhum", "Garhwa", "Giridih", "Godda", "Gumla", "Hazaribagh", "Jamtara", "Khunti", "Koderma", "Latehar", "Lohardaga", "Pakur", "Palamu", "Ramgarh", "Ranchi", "Sahibganj", "Saraikela Kharsawan", "Simdega", "West Singhbhum"],
    "Karnataka": ["Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", "Bidar", "Chamarajanagar", "Chikballapur", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada", "Davangere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar", "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru", "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir"],
    "Kerala": ["Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod", "Kollam", "Kottayam", "Kozhikode", "Malappuram", "Palakkad", "Pathanamthitta", "Thiruvananthapuram", "Thrissur", "Wayanad"],
    "Madhya Pradesh": ["Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani", "Betul", "Bhind", "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas", "Dhar", "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad", "Indore", "Jabalpur", "Jhabua", "Katni", "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh", "Ujjain", "Umaria", "Vidisha"],
    "Maharashtra": ["Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"],
    "Manipur": ["Bishnupur", "Churachandpur", "Chandel", "Imphal East", "Imphal West", "Senapati", "Tamenglong", "Thoubal", "Ukhrul"],
    "Meghalaya": ["East Garo Hills", "East Jaintia Hills", "East Khasi Hills", "North Garo Hills", "Ri Bhoi", "South Garo Hills", "South West Garo Hills", "South West Khasi Hills", "West Garo Hills", "West Jaintia Hills", "West Khasi Hills"],
    "Mizoram": ["Aizawl", "Champhai", "Kolasib", "Lawngtlai", "Lunglei", "Mamit", "Saiha", "Serchhip"],
    "Nagaland": ["Dimapur", "Kiphire", "Kohima", "Longleng", "Mokokchung", "Mon", "Peren", "Phek", "Tuensang", "Wokha", "Zunheboto"],
    "Odisha": ["Angul", "Balangir", "Balasore", "Bargarh", "Bhadrak", "Boudh", "Cuttack", "Deogarh", "Dhenkanal", "Gajapati", "Ganjam", "Jagatsinghpur", "Jajpur", "Jharsuguda", "Kalahandi", "Kandhamal", "Kendrapara", "Kendujhar", "Khordha", "Koraput", "Malkangiri", "Mayurbhanj", "Nabarangpur", "Nayagarh", "Nuapada", "Puri", "Rayagada", "Sambalpur", "Sonepur", "Sundargarh"],
    "Punjab": ["Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib", "Fazilka", "Ferozepur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana", "Mansa", "Moga", "Muktsar", "Pathankot", "Patiala", "Rupnagar", "Sahibzada Ajit Singh Nagar", "Sangrur", "Tarn Taran"],
    "Rajasthan": ["Ajmer", "Alwar", "Banswara", "Baran", "Barmer", "Bharatpur", "Bhilwara", "Bikaner", "Bundi", "Chittorgarh", "Churu", "Dausa", "Dholpur", "Dungarpur", "Hanumangarh", "Jaipur", "Jaisalmer", "Jalore", "Jhalawar", "Jhunjhunu", "Jodhpur", "Karauli", "Kota", "Nagaur", "Pali", "Pratapgarh", "Rajsamand", "Sawai Madhopur", "Sikar", "Sirohi", "Tonk", "Udaipur"],
    "Sikkim": ["East Sikkim", "North Sikkim", "South Sikkim", "West Sikkim"],
    "Tamil Nadu": ["Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kancheepuram", "Karur", "Krishnagiri", "Madurai", "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi", "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli", "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar"],
    "Telangana": ["Adilabad", "Bhadradri Kothagudem", "Hyderabad", "Jagtial", "Jangaon", "Jayashankar Bhupalpally", "Jogulamba Gadwal", "Kamareddy", "Karimnagar", "Khammam", "Komaram Bheem Asifabad", "Mahabubabad", "Mahbubnagar", "Mancherial", "Medak", "Medchal", "Nagarkurnool", "Nalgonda", "Nirmal", "Nizamabad", "Peddapalli", "Rajanna Sircilla", "Rangareddy", "Sangareddy", "Siddipet", "Suryapet", "Vikarabad", "Wanaparthy", "Warangal Rural", "Warangal Urban", "Yadadri Bhuvanagiri"],
    "Tripura": ["Dhalai", "Gomati", "Khowai", "North Tripura", "Sepahijala", "South Tripura", "Unakoti", "West Tripura"],
    "Uttar Pradesh": ["Agra", "Aligarh", "Ambedkar Nagar", "Amethi", "Amroha", "Auraiya", "Azamgarh", "Baghpat", "Bahraich", "Ballia", "Balrampur", "Banda", "Barabanki", "Bareilly", "Basti", "Bhadohi", "Bijnor", "Bulandshahr", "Chandauli", "Chitrakoot", "Deoria", "Etah", "Etawah", "Faizabad", "Farrukhabad", "Fatehpur", "Firozabad", "Gautam Buddha Nagar", "Ghaziabad", "Ghazipur", "Gonda", "Gorakhpur", "Hamirpur", "Hapur", "Hardoi", "Hathras", "Jalaun", "Jaunpur", "Jhansi", "Kannauj", "Kanpur Dehat", "Kanpur Nagar", "Kasganj", "Kaushambi", "Kushinagar", "Lakhimpur Kheri", "Lalitpur", "Lucknow", "Maharajganj", "Mahoba", "Mainpuri", "Mathura", "Mau", "Meerut", "Mirzapur", "Moradabad", "Muzaffarnagar", "Pilibhit", "Pratapgarh", "Raebareli", "Rampur", "Saharanpur", "Sambhal", "Sant Kabir Nagar", "Shahjahanpur", "Shamli", "Shravasti", "Siddharthnagar", "Sitapur", "Sonbhadra", "Sultanpur", "Unnao", "Varanasi"],
    "Uttarakhand": ["Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar", "Nainital", "Pauri Garhwal", "Pithoragarh", "Rudraprayag", "Tehri Garhwal", "Udham Singh Nagar", "Uttarkashi"],
    "West Bengal": ["Alipurduar", "Bankura", "Birbhum", "Cooch Behar", "Dakshin Dinajpur", "Darjeeling", "Hooghly", "Howrah", "Jalpaiguri", "Jhargram", "Kalimpong", "Kolkata", "Malda", "Murshidabad", "Nadia", "North 24 Parganas", "Paschim Bardhaman", "Paschim Medinipur", "Purba Bardhaman", "Purba Medinipur", "Purulia", "South 24 Parganas", "Uttar Dinajpur"],
    "Andaman and Nicobar Islands": ["Nicobar", "North and Middle Andaman", "South Andaman"],
    "Chandigarh": ["Chandigarh"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Dadra and Nagar Haveli", "Daman", "Diu"],
    "Delhi": ["Central Delhi", "East Delhi", "New Delhi", "North Delhi", "North East Delhi", "North West Delhi", "Shahdara", "South Delhi", "South East Delhi", "South West Delhi", "West Delhi"],
    "Jammu and Kashmir": ["Anantnag", "Bandipora", "Baramulla", "Budgam", "Doda", "Ganderbal", "Jammu", "Kathua", "Kishtwar", "Kulgam", "Kupwara", "Poonch", "Pulwama", "Rajouri", "Ramban", "Reasi", "Samba", "Shopian", "Srinagar", "Udhampur"],
    "Ladakh": ["Kargil", "Leh"],
    "Lakshadweep": ["Agatti", "Amini", "Andrott", "Bitra", "Chetlat", "Kavaratti", "Kadmat", "Kalpeni", "Minicoy"],
    "Puducherry": ["Karaikal", "Mahe", "Puducherry", "Yanam"]
}

states = sorted(list(districts_data.keys()))
all_districts = []
for state_districts in districts_data.values():
    all_districts.extend(state_districts)
all_districts = sorted(list(set(all_districts)))  # Remove duplicates and sort

# Prepare datasets for label encoding
le_state = LabelEncoder().fit(states)
le_district = LabelEncoder().fit(all_districts)
seasons = ['Kharif', 'Rabi', 'Summer']
crops = ['Rice', 'Wheat', 'Maize']
le_season = LabelEncoder().fit(seasons)
le_crop = LabelEncoder().fit(crops)

# Dummy model: RandomForestRegressor trained on random data (replace with your trained model)
np.random.seed(42)
X_dummy = np.random.rand(100, 16)
y_dummy = np.random.rand(100)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_dummy, y_dummy)

# Pydantic models for request/response validation

class LocationRequest(BaseModel):
    location: str = Field(..., example="Hyderabad, India")

class CoordinatesResponse(BaseModel):
    latitude: float
    longitude: float

class WeatherRequest(BaseModel):
    latitude: float
    longitude: float
    api_key: str

class WeatherResponse(BaseModel):
    temperature_celsius: float
    humidity_percent: int
    rainfall_mm: Optional[float] = None

class SoilRequest(BaseModel):
    latitude: float
    longitude: float

class SoilResponse(BaseModel):
    ph: Optional[float] = None
    clay_percent: Optional[float] = None
    soil_type: Optional[str] = None
    nitrogen_kg_per_ha: Optional[float] = None

class PredictionRequest(BaseModel):
    state: str
    district: str
    crop_year: int
    season: str
    crop: str
    area: float = Field(..., ge=1.0, le=100.0)
    soil_moisture: float = Field(..., ge=20.0, le=60.0)
    temperature: float = Field(..., ge=15.0, le=35.0)
    humidity: float = Field(..., ge=40.0, le=80.0)
    ph: float = Field(..., ge=5.0, le=8.0)
    rainfall: float = Field(..., ge=100.0, le=500.0)
    nutrient_n: float = Field(..., ge=50.0, le=150.0)
    nutrient_p: float = Field(..., ge=20.0, le=60.0)
    nutrient_k: float = Field(..., ge=50.0, le=200.0)
    pest_level: int = Field(..., ge=0, le=4)
    disease_level: int = Field(..., ge=0, le=4)

class PredictionResponse(BaseModel):
    predicted_yield_kg_per_ha: float
    total_production_kg: float
    prediction_uncertainty_percent: float

class StatesResponse(BaseModel):
    states: list[str]

class DistrictsResponse(BaseModel):
    state: str
    districts: list[str]

# Endpoint: Get list of states
@app.get("/api/states", response_model=StatesResponse)
def get_states():
    return StatesResponse(states=states)

# Endpoint: Get list of districts for a state
@app.get("/api/districts/{state}", response_model=DistrictsResponse)
def get_districts(state: str):
    if state not in districts_data:
        raise HTTPException(status_code=404, detail="State not found")
    return DistrictsResponse(state=state, districts=districts_data[state])

# Endpoint: Get coordinates from location string
@app.post("/api/get-coordinates", response_model=CoordinatesResponse)
def get_coordinates(req: LocationRequest):
    geolocator = Nominatim(user_agent="agripredict_ai_backend")
    location = geolocator.geocode(req.location)
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    return CoordinatesResponse(latitude=location.latitude, longitude=location.longitude)

# Endpoint: Fetch weather data from OpenWeatherMap
@app.post("/api/fetch-weather", response_model=WeatherResponse)
def fetch_weather(req: WeatherRequest):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={req.latitude}&lon={req.longitude}&appid={req.api_key}"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to fetch weather data")
    data = resp.json()
    temp_c = data['main']['temp'] - 273.15
    humidity = data['main']['humidity']
    rainfall = None
    if 'rain' in data and '1h' in data['rain']:
        # Approximate annual rainfall from hourly rain (very rough)
        rainfall = data['rain']['1h'] * 8760
    return WeatherResponse(temperature_celsius=temp_c, humidity_percent=humidity, rainfall_mm=rainfall)

# Endpoint: Fetch soil data from ISRIC SoilGrids
@app.post("/api/fetch-soil", response_model=SoilResponse)
def fetch_soil(req: SoilRequest):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={req.longitude}&lat={req.latitude}&property=phh2o&property=clay&property=nitrogen&depth=0-5cm&value=mean"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to fetch soil data")
    data = resp.json()
    layers = data.get('properties', {}).get('layers', [])
    ph_mean = None
    clay_mean = None
    nitrogen_mean = None
    for layer in layers:
        if layer['name'] == 'phh2o':
            ph_mean = layer['depths'][0]['values']['mean'] / 10.0  # scale as per API docs
        elif layer['name'] == 'clay':
            clay_mean = layer['depths'][0]['values']['mean'] / 10.0
        elif layer['name'] == 'nitrogen':
            nitrogen_mean = layer['depths'][0]['values']['mean'] / 100.0
    soil_type = None
    if clay_mean is not None:
        if clay_mean > 35:
            soil_type = 'Black Soil'
        elif clay_mean > 20:
            soil_type = 'Alluvial Soil'
        elif clay_mean > 10:
            soil_type = 'Red Soil'
        else:
            soil_type = 'Sandy Soil'
    nitrogen_kg_ha = nitrogen_mean * 10 if nitrogen_mean else None  # arbitrary scaling
    return SoilResponse(ph=ph_mean, clay_percent=clay_mean, soil_type=soil_type, nitrogen_kg_per_ha=nitrogen_kg_ha)

# Endpoint: Predict crop yield
@app.post("/api/predict-yield", response_model=PredictionResponse)
def predict_yield(req: PredictionRequest):
    # Validate state and district
    if req.state not in districts_data:
        raise HTTPException(status_code=400, detail=f"Invalid state: {req.state}")
    if req.district not in districts_data[req.state]:
        raise HTTPException(status_code=400, detail=f"Invalid district: {req.district} for state: {req.state}")
    
    # Validate season and crop
    if req.season not in seasons:
        raise HTTPException(status_code=400, detail=f"Invalid season: {req.season}. Must be one of {seasons}")
    if req.crop not in crops:
        raise HTTPException(status_code=400, detail=f"Invalid crop: {req.crop}. Must be one of {crops}")

    # Encode categorical variables
    state_enc = le_state.transform([req.state])[0]
    district_enc = le_district.transform([req.district])[0]
    season_enc = le_season.transform([req.season])[0]
    crop_enc = le_crop.transform([req.crop])[0]

    # Prepare feature vector in correct order
    features = np.array([[
        state_enc,
        district_enc,
        req.crop_year,
        season_enc,
        crop_enc,
        req.area,
        req.soil_moisture,
        req.temperature,
        req.humidity,
        req.ph,
        req.nutrient_n,
        req.nutrient_p,
        req.nutrient_k,
        req.pest_level,
        req.disease_level,
        req.rainfall
    ]])

    # Predict with all trees for uncertainty estimate
    tree_preds = np.array([tree.predict(features)[0] for tree in model.estimators_])
    pred_yield = np.mean(tree_preds)
    pred_production = pred_yield * req.area
    uncertainty = np.std(tree_preds) / pred_yield * 100 if pred_yield != 0 else 0.0

    return PredictionResponse(
        predicted_yield_kg_per_ha=round(pred_yield, 2),
        total_production_kg=round(pred_production, 2),
        prediction_uncertainty_percent=round(uncertainty, 2)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed port here