import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()


# Enable CORS for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model (ensure the correct path)
model_path = '../data/sales_prediction_deeplearning_model_20240922_122652.h5'
model = load_model(model_path, compile=False)

# Load the scalers (you need to save these scalers during your training phase)
scaler_X = MinMaxScaler(feature_range=(-1, 1))  # Load your trained scaler for input data
scaler_y = MinMaxScaler(feature_range=(-1, 1))  # Load your trained scaler for output data

# Define input data schema using Pydantic
class PredictionInput(BaseModel):
    features: list[float]

# API root endpoint (health check)
@app.get("/")
def read_root():
    return {"message": "ML Model Serving API is up and running"}
# Load real sales data (for example, from your dataset)
real_sales_data = np.array([658.0, -4364.0, 6102.0, -1091.0, -229.0, 238.0, 243.0]).reshape(-1, 1)  # Replace this with your actual sales data

# Fit the scaler_y on the actual sales data
scaler_y.fit(real_sales_data)


# Load the data once at the start
try:
    df = pd.read_csv("../data/rossmann/store.csv")  # Ensure the correct path to your CSV file
    stores = df['Store'].unique().tolist()  # Get unique store IDs
except Exception as e:
    print(f"Error loading data: {str(e)}")
    stores = [1,2,3,4,5]


# API endpoint to fetch stores
@app.get("/stores/")
def get_stores():
    try:
        if not stores:
            raise HTTPException(status_code=404, detail="No stores found")
        return {"stores": stores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

# Endpoint to make multi-step predictions (e.g., 42 days or 6 weeks ahead)
@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        # Function to predict future sales using the LSTM model
       
        print(input_data)  # Print to verify structure
        def predict_future_sales(model, last_n_days, scaler_y, forecast_horizon=42):
            future_sales = []
            current_input = last_n_days  # Start with the last N days

            # Ensure the input is in the shape (1, time_steps, 1)
            current_input = current_input.reshape(1, -1, 1)

            for _ in range(forecast_horizon):
                # Predict the next day's sales (scaled)
                next_pred_scaled = model.predict(current_input)

                # Ensure that next_pred_scaled has the correct shape for inverse_transform (should be 2D)
                next_pred_scaled = next_pred_scaled.reshape(-1, 1)

                # Invert scaling to get the actual sales
                next_pred = scaler_y.inverse_transform(next_pred_scaled)

                # Append the predicted sales
                future_sales.append(next_pred[0][0])

                # Update the input: drop the first day and add the new predicted day
                next_pred_scaled = next_pred_scaled.reshape(1, 1, 1)
                current_input = np.append(current_input[:, 1:, :], next_pred_scaled, axis=1)

            return np.array(future_sales)  # Return the future predictions
  

        # Main function to run the prediction process
        def main(model, scaler_y, input_data, forecast_horizon=42):
            # Ensure the input data is a NumPy array
            input_data = np.array(input_data)

            # Reshape the input data to match the model's input expectations
            input_data = input_data.reshape(1, len(input_data), 1)

            # Predict the future sales iteratively
            predicted_sales = predict_future_sales(model, input_data, scaler_y, forecast_horizon=forecast_horizon)

            return predicted_sales

        # Example usage:
        predicted_sales = main(model, scaler_y, input_data.features)  # Use the input data from the request

        return {"predictions": predicted_sales.tolist()}

    except Exception as e:
        # Log the full exception details for debugging
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error occurred: {str(e)}")