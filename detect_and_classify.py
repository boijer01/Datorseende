import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

yolo_model = YOLO('./best_old.pt')
classifier = load_model('./snowboard_skidor_model.keras')
video_path = 'ski.mp4'

class_labels = ['skier', 'snowboarder']

cap = cv2.VideoCapture(video_path)

# Loop over each frame of the video
while cap.isOpened():
    # Read a frame from the video and if no frame it end the loop
    ret, frame = cap.read()
    if not ret: 
        break

    # Detection using YOLO on the current frame
    detections = yolo_model(frame, conf=0.6)[0]

    # Loop for each detected object
    for detection in detections.boxes.data.cpu().numpy():
        # Bounding box coordinates and confidence score from detection
        x1, y1, x2, y2, conf, cls = detection

        # Convertion to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Crop detected object from the frame
        cropped_img = frame[y1:y2, x1:x2]

        if cropped_img.size == 0: # Skip if empty
            continue

        # Process cropped image for classification model
        cropped_img_resized = cv2.resize(cropped_img, (224, 224)) / 255.0
        cropped_img_resized = np.expand_dims(cropped_img_resized, axis=0)

        # Predicts Snowboarder or Skier
        preds = classifier.predict(cropped_img_resized, verbose=0)
        class_idx = np.argmax(preds)
        label = class_labels[class_idx]
        color = (255, 0, 0) if label == 'snowboarder' else (0, 0, 255)

        # Draw rectangle and add label to frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the frame with predictions
    cv2.imshow('Detection', frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Closes when the video ends
cap.release()
cv2.destroyAllWindows()
