from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from geopy.geocoders import Nominatim

# Define the Pydantic model for input validation
class CropDetails(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float 
    ph: float
    rainfall: float

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and label encoder
model = joblib.load('CropModel.pkl')
encoder = joblib.load('ordinalencoder.pkl')


# Root endpoint
@app.get('/')
def runnerfun():
    return {'message': 'The Backend is running properly'}

# Prediction endpoint
@app.post('/predict')
async def predict_crop(details: CropDetails):
    # Convert input data to numpy array
    input_data = np.array([[
        details.N,
        details.P,
        details.K,
        details.temperature,
        details.humidity,
        details.ph,
        details.rainfall
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Reshape prediction to 2D array for inverse_transform
    prediction = prediction.reshape(1, -1)
    
    # Decode the predicted label
    predicted_label = encoder.inverse_transform(prediction)[0]
    predicted_label = str(predicted_label)
    return {'predicted_crop': predicted_label}


@app.post('/location')
def location_data(latitude:float, longitude:float):
    geolocator = Nominatim(user_agent='geo_locator')
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    # print(location)
    return location


#print(location)