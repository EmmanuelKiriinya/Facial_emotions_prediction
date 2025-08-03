# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np

from app.utils.preprocess import preprocess_image

# ------------------------
# Initialize FastAPI
# ------------------------
app = FastAPI(title="Emotion Detection API")

# Enable CORS (for frontend -> backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL for security (e.g., ["https://emotion-frontend.vercel.app"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Load Model
# ------------------------
MODEL_PATH = r"app/models/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (ensure order matches training labels)
CLASS_NAMES = ["angry", "happy", "neutral", "sad", "surprised"]

@app.get("/")
def root():
    return {"message": "Emotion Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    # Predict
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    return {
        "class": CLASS_NAMES[class_idx],
        "confidence": confidence
    }
