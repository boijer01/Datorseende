import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Enkel spårningsklass baserat på centroid-matching
class SimpleTracker:
    def __init__(self, distance_threshold=50):
        self.next_id = 0
        # tracks: id -> {'centroid': (x, y), 'bbox': (x1, y1, x2, y2), 'label': label, 'counted': bool}
        self.tracks = {}
        self.distance_threshold = distance_threshold

    def update(self, detections):
        """
        Parametrar:
            detections: lista med tuple ((x1, y1, x2, y2), label)
        Returnerar:
            uppdaterad dictionary med spårade objekt.
        """
        updated_tracks = {}
        used_detections = set()
        
        # Försök matcha befintliga spår med nya detektioner baserat på avståndet mellan centroider.
        for track_id, track in self.tracks.items():
            prev_centroid = track['centroid']
            best_match = None
            best_dist = float('inf')
            best_det = None
            for i, (bbox, label) in enumerate(detections):
                if i in used_detections:
                    continue
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                dist_val = np.linalg.norm(np.array([cx, cy]) - np.array(prev_centroid))
                if dist_val < best_dist:
                    best_dist = dist_val
                    best_match = (cx, cy, bbox, label)
                    best_det = i
            if best_match is not None and best_dist < self.distance_threshold:
                cx, cy, bbox, label = best_match
                updated_tracks[track_id] = {
                    'centroid': (cx, cy),
                    'bbox': bbox,
                    'label': label,
                    'counted': track['counted']
                }
                used_detections.add(best_det)
        # Registrera nya detektioner som inte matchade befintliga spår
        for i, (bbox, label) in enumerate(detections):
            if i not in used_detections:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                updated_tracks[self.next_id] = {
                    'centroid': (cx, cy),
                    'bbox': bbox,
                    'label': label,
                    'counted': False
                }
                self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks

def point_inside_zone(point, zone):
    """Returnerar True om en punkt (x,y) ligger inom en rektangel definierad av zone=(x1,y1,x2,y2)"""
    x, y = point
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2

# Laddar modeller
yolo_model = YOLO('./best.pt')
classifier = load_model('./snowboard_skidor_model.keras')
class_labels = ['skier', 'snowboarder']

video_path = 'ski.mp4'
cap = cv2.VideoCapture(video_path)

# Initiera tracker och räknare
tracker = SimpleTracker(distance_threshold=50)
skier_count = 0
snowboarder_count = 0
detection_zone = None  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if detection_zone is None:
        frame_height, frame_width = frame.shape[:2]
        detection_zone = (frame_width * 3 // 4, frame_height * 3 // 4, frame_width - 50, frame_height - 50)


    detections_result = yolo_model(frame, conf=0.6)[0]
    detections_list = []
    
    for detection in detections_result.boxes.data.cpu().numpy():
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

        detections_list.append(((x1, y1, x2, y2), label))

    tracks = tracker.update(detections_list)
    
    cv2.rectangle(frame, (detection_zone[0], detection_zone[1]),
                  (detection_zone[2], detection_zone[3]), (128, 128, 128), 2)
    
    for track_id, track in tracks.items():
        (x1, y1, x2, y2) = track['bbox']
        label = track['label']
        centroid = track['centroid']
        color = (255, 0, 0) if label == 'snowboarder' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if not track['counted'] and point_inside_zone(centroid, detection_zone):
            if label == 'snowboarder':
                snowboarder_count += 1
            else:
                skier_count += 1
            track['counted'] = True
            
    cv2.putText(frame, f'Skier Count: {skier_count}', (10, frame_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Snowboarder Count: {snowboarder_count}', (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Optimized Detection & Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
