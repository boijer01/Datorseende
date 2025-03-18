import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

#alla paths, se till att de är rätt c:
yolo_model = YOLO('./best.pt')
classifier = load_model('snowboard_skidor_model.keras')
video_path = 'ski.mp4'

class_labels = ['skier', 'snowboarder']

cap = cv2.VideoCapture(video_path)

count_skier = 0
count_snowboarder = 0
counted_ids = set()

#Hämtar videons storlek för att placera detection-zonen korrekt
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#Skapa detection-zonen, med vår ski.mp4 video så borde den ligga på rätt ställe
detection_zone = (5, frame_height - 200, frame_width - 5, frame_height - 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  
    
    #Kör YOLO tracking på den aktuella ramen
    results = yolo_model.track(frame, tracker='bytetrack.yaml', conf=0.6, persist=True)
    
    #Kontrollera att det finns giltiga detektioner med ID
    if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
        # Extrahera information från detektionerna
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        #går igenom varje detekterad box
        for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
            # Omvandla boxkoordinater till heltal
            x1, y1, x2, y2 = map(int, box)
            
            #Beskär objektet från ramen
            cropped_image = frame[y1:y2, x1:x2]
            if cropped_image.size == 0:
                continue
            
            #Förbereder bilden för klassificeringen
            resized_image = cv2.resize(cropped_image, (224, 224)) / 255.0
            resized_image = np.expand_dims(resized_image, axis=0)
            
            #Utför klassificering
            predictions = classifier.predict(resized_image, verbose=0)
            class_index = np.argmax(predictions)
            label = class_labels[class_index]
            
            color = (255, 0, 0) if label == 'snowboarder' else (0, 0, 255)
            
            #Rita bounding box och text på bilden
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            #Beräkna mittpunkten av boxen
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            #Kolla om mittpunkten ligger inom den fördefinierade detektionszonen
            zone_x1, zone_y1, zone_x2, zone_y2 = detection_zone
            if (zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2 
                    and track_id not in counted_ids):
                counted_ids.add(track_id)
                if label == 'skier':
                    count_skier += 1
                elif label == 'snowboarder':
                    count_snowboarder += 1

        
    #Ritar ut detection zone
    cv2.rectangle(frame, (detection_zone[0], detection_zone[1]),
                  (detection_zone[2], detection_zone[3]), (128, 128, 128), 2)
    
    #Visa räknare på skärmen, borde vara högst upp till vänster
    cv2.putText(frame, f'Skier: {count_skier}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Snowboarder: {count_snowboarder}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    #Visa resultatet live
    cv2.imshow("Tracking, Classification & Counting", frame)
        
    #Vänta på knapptryck (1 ms delay för real-time uppspelning)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()