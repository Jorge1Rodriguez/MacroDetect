import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
import torch

# Diccionario de clases
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
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

def transform_image(image_pil):
    image_np = np.array(image_pil)
    transformed = transform(image=image_np)
    return transformed["image"].unsqueeze(0), image_pil

def predict_mask(model, input_tensor, device):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    percentages = {}
    total_pixels = (prediction != 0).sum()
    for class_id in range(1, 6):
        count = (prediction == class_id).sum()
        if count > 0:
            percentages[macro_names[class_id]] = round((count / total_pixels) * 100, 2)

    return prediction, percentages
