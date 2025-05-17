from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils.processing import UNetResNet50, process_image
import torch
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetResNet50().to(device)
model.load_state_dict(torch.load("backend/model/unet_resnet50.pth", map_location=device))
model.eval()



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        mask_base64, percentages = process_image(image_bytes, model, device)
        return JSONResponse(content={
            "mask": mask_base64,
            "percentages": percentages
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
