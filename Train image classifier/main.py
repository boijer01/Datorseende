import os
import cv2
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Bild storlek
IMG_WIDTH, IMG_HEIGHT = 224, 224


#Läser in annoteringsfiler och matchar dem med bilder
def load_annotations(labels_dir, dataset_dir):
    annotations = []
    for file in os.listdir(labels_dir):
        if file.endswith('.txt'):
            base = os.path.splitext(file)[0]
            # Prova .jpg och .png
            img_path = next((os.path.join(dataset_dir, base+ext) for ext in [".jpg", ".png"] 
                             if os.path.exists(os.path.join(dataset_dir, base+ext))), None)
            if img_path:
                with open(os.path.join(labels_dir, file)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            label = "snowboard" if parts[0] == "0" else "skidor" if parts[0] == "1" else parts[0]
                            annotations.append((img_path, label, *map(float, parts[1:])))
    return annotations


#beskär och skalar bilder baserat på annoteringar
def process_data(annotations):
    imgs, labs = [], []
    for img_path, label, x_center, y_center, width, height in annotations:
        img = cv2.imread(img_path)
        if img is None: 
            continue
        h, w, _ = img.shape
        cx, cy = int(x_center * w), int(y_center * h)
        bw, bh = int(width * w), int(height * h)
        x1, y1 = max(cx - bw // 2, 0), max(cy - bh // 2, 0)
        x2, y2 = min(cx + bw // 2, w), min(cy + bh // 2, h)
        if x1 < x2 and y1 < y2:
            crop = cv2.resize(img[y1:y2, x1:x2], (IMG_WIDTH, IMG_HEIGHT))
            imgs.append(crop)
            labs.append(0 if label.lower() == 'snowboard' else 1)
    return np.array(imgs), np.array(labs)


#Läser in och normaliserar bilder
#Kollar igenom images dataset och dens labels
annotations = load_annotations(r"../NEW SPLIT SYSTEM/labels", r"../NEW SPLIT SYSTEM/dataset") #Ändra denna så det matchar vart du sparat
X, y = process_data(annotations)


#balanserar datasettet så att det är lika många bilder på båda kategorierna
idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
min_count = min(len(idx0), len(idx1))
np.random.seed(42)
selected = np.random.permutation(np.concatenate((
    np.random.choice(idx0, min_count, replace=False),
    np.random.choice(idx1, min_count, replace=False)
)))
X, y = X[selected].astype("float32") / 255.0, to_categorical(y[selected], 2)


#separerar träningsdatan mellan training och test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#bild kvalicifering
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")
])

#konfigurerar optimerare och förlustfunktion
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


#Tränar modellen<3
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


#sparas som keras modell
model.save("snowboard_skidor_model.keras")
