# app/utils/preprocess.py
import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess uploaded image for model inference.
    - Convert bytes → RGB
    - Resize to target_size
    - Normalize [0, 1]
    - Expand dims for batch
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array