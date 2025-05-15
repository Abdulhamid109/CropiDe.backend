from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from geopy.geocoders import Nominatim
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = [
    "http://localhost:5173",  # your Vite/React frontend

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # you can use ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],              # GET, POST, etc.
    allow_headers=["*"],              # Content-Type, Authorization, etc.
)

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
    if location is None:
        return {'error': 'Location not found'}

    return {
        'address': location.address,
        'latitude': location.latitude,
        'longitude': location.longitude,
        'raw': location.raw  # optional: raw dict info
    }


#print(location)