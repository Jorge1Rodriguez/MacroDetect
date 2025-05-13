# backend/utils/processing.py
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
import io
import matplotlib.pyplot as plt
import base64

macro_names = {
    0: "background",
    1: "proteins",
    2: "carbohydrates",
    3: "fats",
    4: "vegetables",
    5: "other"
}

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

class UNetResNet50(nn.Module):
    def __init__(self, in_channels=3, out_classes=6):
        super(UNetResNet50, self).__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, out_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def process_image(image_bytes, model, device):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    input_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Porcentajes
    total_pixels = (prediction != 0).sum()
    percentages = {}
    for class_id in range(1, 6):
        class_pixels = (prediction == class_id).sum()
        if class_pixels > 0:
            percentages[macro_names[class_id]] = round((class_pixels / total_pixels) * 100, 2)

    # Visualización
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(Image.fromarray(image_np).resize((256, 256)))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    mask_plot = axs[1].imshow(prediction, cmap="tab10", vmin=0, vmax=5)
    axs[1].set_title("Predicted Mask")
    axs[1].axis("off")

    cbar = fig.colorbar(mask_plot, ax=axs[1], ticks=range(6))
    cbar.ax.set_yticklabels([macro_names[i] for i in range(6)])

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()

    return img_base64, percentages
