import torch, torch.nn as nn
from myResNet import ResNet50
from PIL import Image
from torchvision import transforms
from config import *

image_size = 224
device = device
model_path = age_pred_model_path
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50()
state = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(state, strict=True)
model.to(device).eval()

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_age(faces, clamp_to=(0, 60)):
    ages = []
    for i in range(len(faces)):
        x = val_transform(faces[i]).unsqueeze(0).to(device)
        pred_age = model(x).squeeze(1).item()   # linear head outputs years
        if clamp_to:
            lo, hi = clamp_to
            pred_age = max(lo, min(hi, pred_age))
        # print("----", pred_age)
        ages.append(pred_age)    
    return ages

