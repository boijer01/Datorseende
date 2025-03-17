import os
from ultralytics import YOLO
from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr

# Ange sökvägen till din dataset-mapp och dataset.yaml-fil (använd raw-strängar för Windows)
dataset_base = r"G:\Users\Simon\Desktop\PROGRAMMERING\Datorseende\Train YOLO model\dataset"
dataset_yaml_path = os.path.join(dataset_base, "dataset.yaml")

# Om dataset.yaml inte finns, skapa den automatiskt
if not os.path.exists(dataset_yaml_path):
    dataset_yaml_content = f"""train: {dataset_base}\\train\\images
val: {dataset_base}\\val\\images

nc: 2
names: ['skier', 'snowboarder']
"""
    with open(dataset_yaml_path, "w") as f:
        f.write(dataset_yaml_content)
    print("dataset.yaml har skapats!")

# Ladda en förtränad YOLOv11-modell (nano-versionen)
# Om filen "yolov11n.pt" inte finns lokalt, laddas den ner automatiskt
model = YOLO("yolov11n.pt")

# Ange träningsparametrar
epochs = 50              # Antal träningsomgångar
imgsz = 640              # Bildstorlek
batch_size = 16          # Batchstorlek
learning_rate = 0.001    # Inlärningshastighet

# Ange var träningsresultaten ska sparas
output_dir = r"G:\Users\Simon\Desktop\PROGRAMMERING\Datorseende\Train YOLO model\output"

# Starta träningen
model.train(
    data=dataset_yaml_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    lr0=learning_rate,
    project=output_dir,
    name="yolov11_training_run"
)

print("Träningen är startad!")
