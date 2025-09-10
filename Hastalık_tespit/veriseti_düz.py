import os
import shutil
import random

# Kaynak ve hedef klasörler
source_dir = "Data2"
output_base = "dataset_split2"
splits = ['train', 'val', 'test']
split_ratios = [0.7, 0.15, 0.15]  # toplamı 1.0 olmalı

# Rastgelelik için sabit tohum
random.seed(42)

# Sınıfları otomatik olarak al
classes = [cls for cls in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, cls))]

# Hedef klasörleri oluştur
for split in splits:
    for cls in classes:
        split_path = os.path.join(output_base, split, cls)
        os.makedirs(split_path, exist_ok=True)

# Verileri böl ve taşı
for cls in classes:
    src_folder = os.path.join(source_dir, cls)
    images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    random.shuffle(images)
    total = len(images)
    train_end = int(split_ratios[0] * total)
    val_end = train_end + int(split_ratios[1] * total)

    split_files = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, file_list in split_files.items():
        for file in file_list:
            src_path = os.path.join(src_folder, file)
            dst_path = os.path.join(output_base, split, cls, file)
            shutil.copyfile(src_path, dst_path)

print("✅ Veriler başarıyla train/val/test olarak ayrıldı!")


