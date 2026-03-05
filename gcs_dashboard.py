from ultralytics import YOLO
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time

# Window setup
root = tk.Tk()
root.title("UAV Tracker Dashboard")
root.configure(bg="#0d1117")

# Layout
main_frame = tk.Frame(root, bg="#0d1117")
main_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(main_frame, width=640, height=480, bg="black")
canvas.pack(side=tk.LEFT)

right_panel = tk.Frame(main_frame, bg="#161b22", width=200)
right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

tk.Label(right_panel, text="GCS DASHBOARD",
         font=("Consolas", 11, "bold"),
         fg="#58a6ff", bg="#161b22").pack(pady=10)

fps_label = tk.Label(right_panel, text="FPS: --",
                     font=("Consolas", 10), fg="#c9d1d9", bg="#161b22")
fps_label.pack(pady=4)

det_label = tk.Label(right_panel, text="Detections: --",
                     font=("Consolas", 10), fg="#c9d1d9", bg="#161b22")
det_label.pack(pady=4)

track_label = tk.Label(right_panel, text="Tracks: --",
                       font=("Consolas", 10), fg="#c9d1d9", bg="#161b22")
track_label.pack(pady=4)

paused = False

def toggle_pause():
    global paused
    paused = not paused
    pause_btn.config(text="▶ Resume" if paused else "⏸ Pause")

def stop():
    cap.release()
    root.destroy()

pause_btn = tk.Button(right_panel, text="⏸ Pause",
                      font=("Consolas", 10, "bold"),
                      bg="#d29922", fg="white",
                      relief=tk.FLAT, padx=8, pady=4,
                      command=toggle_pause)
pause_btn.pack(pady=6, fill=tk.X)

tk.Button(right_panel, text="⏹ Stop",
          font=("Consolas", 10, "bold"),
          bg="#f85149", fg="white",
          relief=tk.FLAT, padx=8, pady=4,
          command=stop).pack(pady=6, fill=tk.X)

# Variables
cap = cv2.VideoCapture(0)
time.sleep(1)

model = None
tracked_objects = {}
trails = {}
next_id = 0
fps_time = time.time()
frame_count = 0

# Main loop
def update_frame():
    global fps_time, tracked_objects, trails, next_id, model, frame_count

    if model is None:
        model = YOLO("yolov8n.pt")

    if paused:
        root.after(10, update_frame)
        return

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame_count += 1
    if frame_count % 2 == 0:
        root.after(10, update_frame)
        return

    # FPS
    now = time.time()
    fps = 1.0 / (now - fps_time + 0.001)
    fps_time = now

    # YOLO Detection
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    rects = []
    det_count = 0

    for box in boxes:
        conf = float(box.conf[0])
        if conf > 0.45:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = names[int(box.cls[0])]
            rects.append((x1, y1, x2, y2))
            det_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Centroid tracking
    new_tracked = {}
    for (x1, y1, x2, y2) in rects:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        matched_id = None
        min_dist = float("inf")

        for obj_id, (ox, oy) in tracked_objects.items():
            dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
            if dist < min_dist:
                min_dist = dist
                matched_id = obj_id

        if min_dist < 120:
            new_tracked[matched_id] = (cx, cy)
        else:
            new_tracked[next_id] = (cx, cy)
            next_id += 1

    tracked_objects = new_tracked

    # Draw trails and IDs
    for obj_id, (cx, cy) in tracked_objects.items():
        if obj_id not in trails:
            trails[obj_id] = []
        trails[obj_id].append((cx, cy))
        if len(trails[obj_id]) > 20:
            trails[obj_id].pop(0)

        pts = trails[obj_id]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (0, 255, 255), 2)

        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"ID {obj_id}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Update stats
    fps_label.config(text=f"FPS: {fps:.1f}")
    det_label.config(text=f"Detections: {det_count}")
    track_label.config(text=f"Tracks: {len(tracked_objects)}")

    # Show frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    canvas.image = photo

    root.after(10, update_frame)

# Start
update_frame()
root.mainloop()
