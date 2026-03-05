import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import numpy as np

# Centroid Tracker
class CentroidTracker:
    def __init__(self):
        self.next_id = 0          # ID counter, starts at 0
        self.objects = {}         # stores {id: centroid}
        self.disappeared = {}     # stores {id: frames missing}

    def update(self, rects):
        # rects = list of (x1, y1, x2, y2) from YOLO this frame

        # If nothing detected, mark all existing objects as disappeared
        if len(rects) == 0:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > 50:  # remove after 50 frames
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        # Calculate centroid for each detected box
        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            input_centroids.append((cx, cy))

        # If no existing objects, register all as new
        if len(self.objects) == 0:
            for c in input_centroids:
                self.objects[self.next_id] = c
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        else:
            # Match new centroids to existing ones by distance
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

            for i, new_c in enumerate(input_centroids):
                # Find closest existing centroid
                min_dist = float("inf")
                best_id = None

                for j, old_c in enumerate(obj_centroids):
                    dist = np.sqrt((new_c[0]-old_c[0])**2 + (new_c[1]-old_c[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_id = obj_ids[j]

                if min_dist < 120:  # within 80 pixels = same object
                    self.objects[best_id] = new_c
                    self.disappeared[best_id] = 0
                else:
                    # Too far = new object
                    self.objects[self.next_id] = new_c
                    self.disappeared[self.next_id] = 0
                    self.next_id += 1

        return self.objects


# Main code
model = YOLO("yolov8n.pt")
tracker = CentroidTracker()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    # Collect boxes that pass confidence threshold
    rects = []
    for box in boxes:
        conf = float(box.conf[0])
        if conf > 0.45:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            rects.append((x1, y1, x2, y2))

            # Draw the detection box in green
            label = names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Update tracker with this frame's boxes
    tracked = tracker.update(rects)

    # Draw tracker IDs at each centroid
    for obj_id, (cx, cy) in tracked.items():
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"ID {obj_id}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()