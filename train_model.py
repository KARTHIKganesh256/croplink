import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

# Create mock dataset
np.random.seed(42)
data = {
    "rainfall": np.random.randint(100, 800, 200),
    "temp": np.random.uniform(18, 35, 200),
    "fertilizer": np.random.randint(50, 200, 200),
    "soil_pH": np.random.uniform(4.5, 8.0, 200),
    "ndvi": np.random.uniform(0.2, 0.9, 200),
}
data["yield"] = (
    10*data["rainfall"]
    + 50*data["ndvi"]
    + 5*data["fertilizer"]
    - 20*abs(data["soil_pH"] - 6.5)
    + np.random.normal(0, 100, 200)
)

df = pd.DataFrame(data)

X = df.drop(columns=["yield"])
y = df["yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

# Save model to backend folder
joblib.dump(model, "yield_model.pkl")
print("✅ Model trained and saved as yield_model.pkl")
