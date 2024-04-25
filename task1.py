import uvicorn
from fastapi import FastAPI, File, UploadFile
import keras
from keras.preprocessing import image
import numpy as np
import sys
from PIL import Image
import io
from imageio import v3 as iio
from fastapi import Response


app = FastAPI()

# Function to Load the model
def load_model(path: str):
    return keras.models.load_model(path) 

# Function to Predict digit
def predict_digit(model, data_point):
    prediction = model.predict(data_point)
    digit = np.argmax(prediction) # The predicted digit is the argmax of the output of the model
    return str(digit)

# API endpoint
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    model_path = sys.argv[1]  # Get model path from command line argument
    loaded_model = load_model(model_path) # Load the model
    if file.content_type.startswith('image'): # Check if the input file is a image file
        image = await file.read() # Read the input file
        
    else:
        return {"error": "Uploaded file is not an image."}
    
    pil_image = Image.open(io.BytesIO(image)) # Open the image using PIL
    numpy_image = np.array(pil_image) # Convert PIL image to NumPy array
    numpy_image=numpy_image/255.0 # Convert the values to between 0 and 1
    data_point=numpy_image.reshape(1,28*28) # Reshape the image array to create a seralized array of 784 elements
    digit = predict_digit(loaded_model, data_point) # Use the model to predict the digit
    
    return {"digit": digit}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Specify the path of the model as a command line argument and no other arguments apart from that!!!")
        sys.exit(1)
    uvicorn.run(app, host="0.0.0.0", port=8000)
