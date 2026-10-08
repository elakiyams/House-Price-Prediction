import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================
# 1. LOAD DATA
# =====================================

df = pd.read_csv("data/house_data.csv")

print("Data loaded successfully")


# =====================================
# 2. FEATURES AND TARGET
# =====================================

X = df[["Area", "Bedrooms", "Age"]]

y = df["Price"]


# =====================================
# 3. SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================
# 4. CREATE MODEL
# =====================================

model = LinearRegression()


# =====================================
# 5. TRAIN MODEL
# =====================================

model.fit(X_train, y_train)

print("Model training completed")


# =====================================
# 6. PREDICTION
# =====================================

y_pred = model.predict(X_test)


# =====================================
# 7. EVALUATION
# =====================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nModel Results")

print("MAE:", mae)

print("MSE:", mse)

print("R2 Score:", r2)


# =====================================
# 8. SAVE MODEL
# =====================================

import pickle

# Save the trained model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
import joblib

joblib.dump(model, "model/house_price_model.pkl")
