# 😄 Facial Emotion Detection using Deep Learning

## 🔍 Project Overview

This project focuses on building a deep learning model to detect **emotions from facial expressions** using facial images. We utilize the **TensorFlow** and **Keras** libraries for preprocessing and training.

---

## 🛠️ Step 1: Environment Setup and Library Imports

We begin by importing the required Python libraries:

```python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
```
TensorFlow/Keras: For model training and dataset utilities

NumPy and Pandas: For numerical operations and data handling

Matplotlib: For image visualization

## 📁 Step 2: Load Dataset from Directory
The dataset is loaded using image_dataset_from_directory, a high-level utility from TensorFlow. The dataset is assumed to be stored in a directory where subfolders represent emotion classes.

```
batch_size = 32
img_height = 48
img_width = 48

train_ds = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/facial-emotion-dataset/train_dir",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/facial-emotion-dataset/train_dir",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
```
**🔖 Parameters:**
validation_split=0.2: Reserve 20% of the dataset for validation

subset: Indicates whether to load training or validation data

image_size=(48, 48): Resize all images to 48x48 pixels

batch_size=32: Use mini-batches of 32 images

seed=123: Ensures reproducibility of splits

## 🏷️ Step 3: Retrieve Class Labels
Print the emotion class names derived from the subdirectory structure:

```
class_names = train_ds.class_names
print(class_names)
```
_output_  
['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  

## 🖼️ Step 4: Visualize Sample Images
Visual inspection helps verify correct loading and label assignment.

```
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
```
This renders a 3x3 grid of training images with corresponding emotion labels.

## ⚙️ Step 5: Normalize the Dataset
Normalize pixel values from [0, 255] to [0.0, 1.0] to improve model convergence during training.

```
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
```
- Converts image data type to float32

- Scales pixel values to the [0.0, 1.0] range

## ✅ Step 6: Verify Normalization
Check the pixel value range and data type after normalization.

```
for image, label in train_ds.take(1):
    print("Image dtype:", image.dtype)
    print("Min pixel value:", tf.reduce_min(image).numpy())
    print("Max pixel value:", tf.reduce_max(image).numpy())
```