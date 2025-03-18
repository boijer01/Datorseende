import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import subprocess

yolo_model = YOLO('./best_old.pt')
classifier = load_model('./snowboard_skidor_model.keras')
class_labels = ['skier', 'snowboarder']

youtube_url = 'https://www.youtube.com/watch?v=Wr9b5aYA4mI'


stream_url = subprocess.check_output(
    ['streamlink', youtube_url, 'best', '--stream-url']
).decode().strip()

cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise Exception("Cannot open stream URL. Check FFmpeg and Streamlink installations.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Stream buffering or paused, retrying...")
        continue

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
        class_labels = ['skier', 'snowboarder']
        label = class_labels[np.argmax(preds)]
        color = (255, 0, 0) if label == 'snowboarder' else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Livestream Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
