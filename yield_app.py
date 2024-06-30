import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('yield_model.h5')

# Load encoders and scalers
scaler = StandardScaler()
label_encoder_crop = LabelEncoder()
label_encoder_season = LabelEncoder()
label_encoder_state = LabelEncoder()

data = pd.read_csv('crop_yield.csv')
label_encoder_crop.fit(data['Crop'])
label_encoder_season.fit(data['Season'])
label_encoder_state.fit(data['State'])
scaler.fit(data[['Crop_Year', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']])

# Streamlit app
st.title("Crop Yield Prediction")

# Inputs from user
crop = st.selectbox("Select Crop", label_encoder_crop.classes_)
season = st.selectbox("Select Season", label_encoder_season.classes_)
state = st.selectbox("Select State", label_encoder_state.classes_)
crop_year = st.number_input("Crop Year", min_value=1900, max_value=2100, step=1)
area = st.number_input("Area")
production = st.number_input("Production")
annual_rainfall = st.number_input("Annual Rainfall")
fertilizer = st.number_input("Fertilizer")
pesticide = st.number_input("Pesticide")

# Preprocess inputs
crop_encoded = label_encoder_crop.transform([crop])[0]
season_encoded = label_encoder_season.transform([season])[0]
state_encoded = label_encoder_state.transform([state])[0]
input_data = np.array([[crop_year, area, production, annual_rainfall, fertilizer, pesticide]])
input_data_scaled = scaler.transform(input_data)

# Combine all inputs
features = np.array([[input_data_scaled[0, 0], crop_encoded, state_encoded, input_data_scaled[0, 1], 
                      input_data_scaled[0, 2], input_data_scaled[0, 3], input_data_scaled[0, 4]]])

# Reshape for LSTM model
features = features.reshape((features.shape[0], 1, features.shape[1]))

# Predict
if st.button("Predict"):
    prediction = model.predict(features)
    st.write(f"Predicted Yield: {prediction[0][0]}")
