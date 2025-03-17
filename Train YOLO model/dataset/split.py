import os
import shutil
import random

# Ange sökvägen till ditt dataset (använd raw-strängar för Windows-paths)
dataset_base = r"G:\Users\Simon\Desktop\PROGRAMMERING\Datorseende\Train YOLO model\dataset"
images_path = os.path.join(dataset_base, "images")
labels_path = os.path.join(dataset_base, "labels")

# Ange destinationer för träning och validering
train_images_path = os.path.join(dataset_base, "train", "images")
train_labels_path = os.path.join(dataset_base, "train", "labels")
val_images_path = os.path.join(dataset_base, "val", "images")
val_labels_path = os.path.join(dataset_base, "val", "labels")

# Skapa destinationerna om de inte finns
for path in [train_images_path, train_labels_path, val_images_path, val_labels_path]:
    os.makedirs(path, exist_ok=True)

# Hämta alla bildfiler (justera filtillägg om nödvändigt)
image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Blanda filerna slumpmässigt
random.shuffle(image_files)

# Bestäm uppdelningen: t.ex. 80% träning, 20% validering
split_index = int(len(image_files) * 0.8)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Kopiera bilder och motsvarande label-filer till träning och validering
for file in train_files:
    # Kopiera bild
    src_image = os.path.join(images_path, file)
    dst_image = os.path.join(train_images_path, file)
    shutil.copy(src_image, dst_image)
    
    # Hitta och kopiera motsvarande label-fil (antag att label-filen heter samma som bilden men med .txt)
    label_file = os.path.splitext(file)[0] + ".txt"
    src_label = os.path.join(labels_path, label_file)
    if os.path.exists(src_label):
        dst_label = os.path.join(train_labels_path, label_file)
        shutil.copy(src_label, dst_label)

for file in val_files:
    # Kopiera bild
    src_image = os.path.join(images_path, file)
    dst_image = os.path.join(val_images_path, file)
    shutil.copy(src_image, dst_image)
    
    # Kopiera motsvarande label-fil om den finns
    label_file = os.path.splitext(file)[0] + ".txt"
    src_label = os.path.join(labels_path, label_file)
    if os.path.exists(src_label):
        dst_label = os.path.join(val_labels_path, label_file)
        shutil.copy(src_label, dst_label)

print("Dataset har delats upp i train och val!")
