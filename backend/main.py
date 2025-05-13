# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from backend.utils.processing import UNetResNet50, process_image


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetResNet50().to(device)
model.load_state_dict(torch.load("backend/model/unet_resnet50.pth", map_location=device))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_base64, percentages = process_image(image_bytes, model, device)
    return JSONResponse(content={
        "image": image_base64,
        "percentages": percentages
    })
