import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

yolo_model = YOLO('./best.pt')
classifier = load_model('./snowboard_skidor_model.keras')

class_labels = ['skier', 'snowboarder']
video_path = 'ski.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = yolo_model(frame, conf=0.6)[0] 

    for detection in detections.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = detection

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            continue

        cropped_img_resized = cv2.resize(cropped_img, (224, 224)) / 255.0
        cropped_img_resized = np.expand_dims(cropped_img_resized, axis=0)

        preds = classifier.predict(cropped_img_resized, verbose=0)
        class_idx = np.argmax(preds)
        label = class_labels[class_idx]

        color = (255, 0, 0) if label == 'snowboarder' else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Optimized Detection & Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
