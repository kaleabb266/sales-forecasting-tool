# sales_prediction.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# Data Preprocessing Functions
def load_data(file_path):
    return pd.read_csv(file_path)

def feature_engineering(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Ensure 'Date' is in datetime format, coerce errors
    if df['Date'].isnull().any():
        print("Warning: Some dates could not be parsed and will be set to NaT.")
    df['Weekday'] = df['Date'].dt.weekday
    df['IsWeekend'] = df['Weekday'] >= 5
    df['MonthPart'] = df['Date'].dt.day.apply(lambda x: 'Start' if x <= 10 else 'Mid' if x <= 20 else 'End')
    df['CompetitionOpenSince'] = pd.to_datetime(df.apply(lambda x: f"{int(x['CompetitionOpenSinceYear'])}-{int(x['CompetitionOpenSinceMonth'])}-01", axis=1))
    df['Promo2Since'] = pd.to_datetime(df.apply(lambda x: f"{int(x['Promo2SinceYear'])}-W{x['Promo2SinceWeek']}-1", axis=1).str.replace('-W', '-W0'))
    df['DaysSinceCompetitionOpen'] = (df['Date'] - df['CompetitionOpenSince']).dt.days
    df['DaysSincePromo2Start'] = (df['Date'] - df['Promo2Since']).dt.days
    df['IsHoliday'] = df['StateHoliday'].apply(lambda x: 1 if x in ['a', 'b', 'c'] else 0)
    return df


def scale_data(df, scaler=None):
    if not scaler:
        scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    return scaled_features, scaler

# Model Building Functions
def build_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    return pipeline.score(X_test, y_test)

# Serialization
def serialize_model(model, filename):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    joblib.dump(model, f"{filename}-{timestamp}.pkl")

# # Deep Learning Functions
# def build_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=input_shape))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     return model

# def train_lstm_model(model, X_train, y_train, epochs=10, batch_size=32):
#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
#     return model

# # Utility Functions
# def plot_feature_importance(model, features):
#     importance = model.feature_importances_
#     sns.barplot(x=importance, y=features)
#     plt.xlabel('Feature Importance')
#     plt.ylabel('Feature')
#     plt.title('Feature Importance Visualization')
#     plt.show()

# def estimate_confidence_interval(predictions, confidence_level=0.95):
#     mean = np.mean(predictions)
#     std_err = np.std(predictions) / np.sqrt(len(predictions))
#     margin_of_error = std_err * 1.96  # 95% confidence
#     return mean - margin_of_error, mean + margin_of_error
