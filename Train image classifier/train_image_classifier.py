import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

IMG_WIDTH, IMG_HEIGHT = 224, 224

#Paths till dataset
LABELS_DIR = "../dataset/labels" 
DATASET_DIR = "../dataset/"

#laddar annoteringar och matchar dem med bilder
def load_annotations(labels_dir, dataset_dir):
    annotations = []
    for file in os.listdir(labels_dir):
        if file.endswith('.txt'):
            base = os.path.splitext(file)[0]
            img_path = next((os.path.join(dataset_dir, base+ext) for ext in [".jpg", ".png"]
                             if os.path.exists(os.path.join(dataset_dir, base+ext))), None)
            if img_path:
                with open(os.path.join(labels_dir, file)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            label = 0 if parts[0] == "0" else 1  # 0 = Snowboard, 1 = Skidor
                            annotations.append((img_path, label, *map(float, parts[1:])))
    return annotations

#beskär och förbereder bilder baserat på annoteringar
def process_data(annotations):
    imgs, labs = [], []
    for img_path, label, x_center, y_center, width, height in annotations:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Kunde inte läsa bilden {img_path} :( men fortsätter ändå")
            continue

        h, w, _ = img.shape

        # Beräkna koordinater
        x1 = max(int((x_center - width / 2) * w), 0)
        y1 = max(int((y_center - height / 2) * h), 0)
        x2 = min(int((x_center + width / 2) * w), w)
        y2 = min(int((y_center + height / 2) * h), h)

        if x1 >= x2 or y1 >= y2:
            print(f"VARNING: Ogiltig beskärning för {img_path} Skippas")
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"VARNING: Tom bild efter beskärning för {img_path} Skippas")
            continue

        crop = cv2.resize(crop, (IMG_WIDTH, IMG_HEIGHT))
        imgs.append(crop)
        labs.append(label)

    return np.array(imgs), np.array(labs)

#Ladda datasetet
annotations = load_annotations(LABELS_DIR, DATASET_DIR)
X, y = process_data(annotations)

X = X.astype("float32") / 255.0
y = to_categorical(y, 2)

#Dela upp datasetet i träning och test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------
# Tränar med MOBILENETV2 själva grundmodlen
# -------------------------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Fryser basmodellen

#Anpassa modellen för snowboard/skidor-klassificering
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(2, activation="softmax")(x)  #Två klasser

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

#Träna modellen (Transfer Learning)
print("Tränar modellen med fryst basmodell...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# -------------------------------------------
# Fine-Tuning
# -------------------------------------------
print("Aktiverar fine-tuning...")
for layer in base_model.layers[-20:]:
    layer.trainable = True

#Kompilera om med lägre inlärningshastighet
model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

#Spara modellen
model.save("snowboard_skidor_model.keras")
