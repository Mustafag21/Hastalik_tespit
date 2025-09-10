from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Geliştirme için tüm kaynaklara izin, production'da kısıtla
    allow_methods=["*"],
    allow_headers=["*"],
)
class_names = ["Akne", "Egzema","Sedef", "Vitiligo"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("resnet50_model2.pth", map_location=device))
model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.95):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        max_prob, pred_idx = torch.max(probs, 1)
    max_prob_value = max_prob.item()
    predicted_label = class_names[pred_idx.item()]
    if max_prob_value < threshold:
        return {
            "message": "❗ Bu görsel bir hastalık sınıfına ait gibi görünmüyor. Lütfen uygun bir cilt görseli yükleyin.",
            "confidence": max_prob_value,
            "predicted_label": None
        }
    else:
        return {
            "message": "✅ Tahmin başarılı",
            "confidence": max_prob_value,
            "predicted_label": predicted_label
        }


