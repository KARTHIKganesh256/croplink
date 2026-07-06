# AgriPredict AI 🌾
> Advanced Crop Yield Analytics & Real-Time AI Prediction Platform

AgriPredict AI is a premium, unified crop yield prediction and agricultural analytics platform. By merging the predictive analytics of a global FAO crop dataset with live geo-telemetry API integrations (weather and soil), the platform enables farmers, agronomists, and researchers to forecast yield in real-time and obtain actionable agronomic advice.

---

## 🌟 Key Features

### 1. Unified Glassmorphic Dashboard
- **Modern UI/UX**: Designed using modern glassmorphic elements, HSL-tailored colors, smooth animations, and a responsive layout.
- **Dark Mode**: Sleek theme selector (Sun/Moon icon in header) to toggle between light and dark visual aesthetics.
- **Responsive Tabs**:
  - **Analytics & Insights**: Direct showcase of all key visualization plots (correlation heatmaps, pairplots, histograms, and geospatial maps).
  - **AI Yield Predictor**: The forecasting panel containing the telemetry controls and interactive forms.

### 2. Live Geo-Telemetry Fetching
- **Address Resolution**: Enter any location (e.g., `"Hyderabad, India"`, `"Cairo, Egypt"`) to resolve latitude and longitude using Nominatim geocoding.
- **Real-Time Weather**: Queries the OpenWeatherMap API to retrieve current local temperature, humidity, and annual rainfall projections.
- **Dynamic Soil Composition**: Queries the international ISRIC SoilGrids database to obtain real-time soil pH, clay percentage, and nitrogen (N) level at a 0-5cm depth.
- **Auto-Fill Fields**: Fills all complex meteorological and chemical inputs in the predictor automatically with one click.

### 3. Dual Machine Learning Engines
- **Global FAO Predictor (Real Pre-trained Model)**:
  - Powered by a high-accuracy Bagging Regressor model (`best_model.pkl`) pre-trained on 28,000+ historical entries.
  - Features: Area (Country), Crop (Item), Year, Average Temperature, Annual Rainfall, and Pesticide application levels (tonnes).
- **Indian State/District Detailed Predictor (Real-time Model)**:
  - Powered by a Random Forest Regressor trained automatically on startup via a custom agronomic dataset.
  - Matches State and District dropdowns linked dynamically from JSON data.
  - Features: State, District, Season, Crop, Field Area, Soil Moisture, Soil pH, Temperature, Humidity, Rainfall, N-P-K Fertilizers, Pest Level, and Disease Level.
  - Displays **Prediction Uncertainty (%)** and **Total Field Production (kg)** metrics.

### 4. Smart Agronomic Advisor
- Generates context-specific, rule-based AI recommendations based on predictions and soil metrics:
  - **Soil pH warnings**: Recommends agricultural lime for acidic soil and elemental sulfur/organic compost for alkaline soil.
  - **Fertilizer recommendations**: Signals nitrogen, phosphorus, or potassium deficiencies and suggests fertilizer ratios (e.g., urea, SSP, MOP).
  - **Pest & Disease alerts**: Prompts active bio-pesticide treatments or foliage pruning for high pest/disease metrics.

### 5. Model Evaluation Gallery
- Interactive showcase comparing 7 different regression models (Bagging Regressors, Random Forests, XGBoost, Decision Trees, KNN, and Linear Regression).
- Highlights **Accuracy (%)** and **R2 scores** for each algorithm, accompanied by actual-vs-predicted comparison scatter plots.

---

## 📂 Project Structure

```
d:/SIH/
├── server.py                   # Unified Flask Web Backend & API server
├── croplink_helper.py          # Indian Crop Model dataset generator and RF training pipeline
├── templates/
│   └── index.html              # Core Glassmorphic HTML5/CSS3/JavaScript dashboard template
├── croplink/
│   ├── states_districts.json   # Linked state and district mapping database for India
│   ├── encoder_mappings.json   # Reconstructs label encoders for global Area/Item variables
│   └── index.html              # (Backup) Original frontend html template
├── best_model.pkl              # 151MB pre-trained global Bagging Regressor model
├── yield_df.csv                # Global FAO crop dataset (28,242 rows)
├── app_unified.log             # System log file mapping coordinates and prediction logs
├── croplink_model.pkl          # Cached Indian crop random forest model
├── croplink_encoders.pkl       # Cached label encoders for Indian crop model categories
├── README.md                   # Full information & startup manual
└── [Visualization PNGs]        # Pre-generated data plots (correlation heatmap, pairplots, etc.)
```

---

## 🛠️ Installation & Setup

### Prerequisites
Make sure Python 3.10+ is installed on your system.

### 1. Install Dependencies
Install all required libraries through `pip`:
```bash
pip install flask pandas numpy scikit-learn geopy joblib requests
```

### 2. Start the Platform
Run the unified Flask server:
```bash
python server.py
```
On startup, the backend will:
1. Reconstruct the global label encoders from `croplink/encoder_mappings.json`.
2. Load the global `best_model.pkl` predictor.
3. Automatically train the Indian Detailed Predictor Random Forest model (if not cached) and store it as `croplink_model.pkl` in the workspace.
4. Launch the web application on `http://localhost:5000`.

### 3. Open the Dashboard
Navigate to your web browser and open:
[http://localhost:5000](http://localhost:5000)

---

## 🚜 How to Use the Predictor

1. Select the **AI Yield Predictor** tab at the top.
2. In the **Geo-Telemetry** section, enter a location (e.g. `Hyderabad, India`) and click **Fetch Live Data**.
3. Live satellite coordinate indicators, weather conditions, and soil data will load instantly into the details box.
4. Select either the **Global Model (FAO)** or **Indian Detailed Model** tab.
5. Watch as the retrieved temperature, rainfall, pH, and nitrogen composition are automatically populated inside the inputs.
6. Make fine-tuned edits (such as specifying field area, pest threat level, or phosphorus/potassium concentrations).
7. Click **Calculate Expected Yield** to run the prediction model.
8. The gauge updates immediately with the forecasted yield, while the AI Advice list logs specialized farming tips based on your soil and crop status.
