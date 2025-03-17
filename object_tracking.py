import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

#alla paths
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
detection_zone = (frame_width - 300, 300, frame_width - 100, frame_height - 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  
    
    #Kör YOLO tracking på den aktuella ramen
    results = yolo_model.track(frame, tracker='bytetrack.yaml', conf=0.6, persist=True)
    
    #Om vi har några detektioner
    if results[0].boxes is not None and hasattr(results[0].boxes, 'id'):
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, track_id, conf, cls) in enumerate(zip(boxes, track_ids, confs, clss)):
                x1, y1, x2, y2 = map(int, box)
                
                #Klassificera objekt
                cropped_img = frame[y1:y2, x1:x2]
                if cropped_img.size == 0:
                    continue
                cropped_img_resized = cv2.resize(cropped_img, (224, 224)) / 255.0
                cropped_img_resized = np.expand_dims(cropped_img_resized, axis=0)
                preds = classifier.predict(cropped_img_resized, verbose=0)
                class_idx = np.argmax(preds)
                label = class_labels[class_idx]
                
                #Välj färg baserat på klass
                color = (255, 0, 0) if label == 'snowboarder' else (0, 0, 255)
                
                #Rita bounding box och text 
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                #Beräkna mittpunkten för objektet
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                #Kolla om mittpunkten är inom detektionszonen och track_id inte räknats
                x1_zone, y1_zone, x2_zone, y2_zone = detection_zone
                if x1_zone <= center_x <= x2_zone and y1_zone <= center_y <= y2_zone and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    if label == 'skier':
                        count_skier += 1
                    elif label == 'snowboarder':
                        count_snowboarder += 1
    
    #Rita ut detection-zonen i grått
    cv2.rectangle(frame, (detection_zone[0], detection_zone[1]),
                  (detection_zone[2], detection_zone[3]), (128, 128, 128), 2)
    
    #Visa räknare på skärmen
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