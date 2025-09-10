import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

# 📁 Veri yolu ve transform
data_root = "dataset_split2"
val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🧪 Test verisini yükle
test_dataset = datasets.ImageFolder(root=f"{data_root}/test", transform=val_test_transforms)
class_names = test_dataset.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔄 Modeli yükle
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("resnet50_model2.pth", map_location=device))
model.to(device)
model.eval()
print("✅ Model yüklendi.")

# ✅ Rastgele 5 görsel göster ve tahmin yap
def show_random_predictions(dataset, model, num_images=5):
    indices = random.sample(range(len(dataset)), num_images)
    with torch.no_grad():
        for idx in indices:
            image, label = dataset[idx]
            input_tensor = image.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred = torch.max(output, 1)

            true_label = class_names[label]
            pred_label = class_names[pred.item()]
            result = "✅ Doğru" if pred_label == true_label else "❌ Yanlış"

            # Görseli göster
            img = image.permute(1, 2, 0).numpy()
            img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            img = np.clip(img, 0, 1)

            plt.imshow(img)
            plt.title(f"Tahmin: {pred_label}\nGerçek: {true_label} → {result}")
            plt.axis('off')
            plt.show()

# 🎯 Modelin genel performansı: Confusion matrix, F1 vs
def evaluate_model(model, dataset):
    all_preds = []
    all_labels = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"F1 Skoru (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("🔍 Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.show()

# ▶️ Çalıştır
show_random_predictions(test_dataset, model, num_images=5)
evaluate_model(model, test_dataset)

