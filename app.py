# app.py
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import tensorflow as tf  # or torch, depending on your model

app = FastAPI()

# Load your trained model
MODEL_PATH = "traffic_sign_model.h5"   # change to your saved model file
model = tf.keras.models.load_model(MODEL_PATH)

# Define preprocessing (must match training preprocessing)
def preprocess_image(image: Image.Image):
    image = image.resize((64, 64))  # adjust size as per training
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess and predict
    input_data = preprocess_image(image)
    preds = model.predict(input_data)
    predicted_class = np.argmax(preds, axis=1)[0]

    return JSONResponse(content={"predicted_class": int(predicted_class)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
