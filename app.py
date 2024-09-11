from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

# Load the model once during app startup
model_path = "G://Chessman//best_chess_model.keras"
model = load_model(model_path)
print(f"Model loaded from: {model_path}")

# Define chess piece categories
categories = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']

IMG_SIZE = 128  # Assuming model was trained on 128x128 images

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))  # Resize to (128, 128)
    img_array = np.array(image) / 255.0  # Normalize to [0, 1] range
    
    # Remove alpha channel if present
    if img_array.shape[-1] == 4:  
        img_array = img_array[..., :3]
    
    return img_array

def predict_chess_piece(image: Image.Image):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    pred = model.predict(img)
    predicted_class = categories[np.argmax(pred)]
    return predicted_class

# Serve static files (CSS) from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML templates
@app.get("/", response_class=HTMLResponse)
async def main():
    with open("templates/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    prediction = predict_chess_piece(img)
    return {"prediction": prediction}
