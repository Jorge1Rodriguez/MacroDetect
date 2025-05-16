import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import base64
import matplotlib.pyplot as plt

macro_names = {
    
    0: "background",
    1: "proteins",
    2: "carbohydrates",
    3: "fats",
    4: "vegetables",
    5: "other"
}

macro_colors = {
    0: "#c0c0c0",  # background
    1: "#1f77b4",  # protein
    2: "#ff7f0e",  # carbohydrate
    3: "#2ca02c",  # fat
    4: "#d62728",  # vegetable
    5: "#e377c2",  # other
}

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

class UNetResNet50(nn.Module):
    def __init__(self, in_channels=3, out_classes=6):
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, out_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def process_image(image_bytes, model, device):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)
    input_tensor = transform(image=np_image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output.squeeze(), dim=0).cpu().numpy()


    total_pixels = (prediction != 0).sum()
    percentages = {}
    for class_id in range(1, 6):
        pixels = (prediction == class_id).sum()
        if pixels > 0:
            percentages[macro_names[class_id]] = round((pixels / total_pixels) * 100, 2)

    rgb_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    for class_id, hex_color in macro_colors.items():
        if class_id == 0:
            continue
        color = tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        rgb_mask[prediction == class_id] = color

    result = Image.fromarray(rgb_mask)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    b64_mask = base64.b64encode(buf.getvalue()).decode("utf-8")

    return b64_mask, percentages
