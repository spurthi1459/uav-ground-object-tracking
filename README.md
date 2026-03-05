# 🚁 UAV Ground Object Detection & Multi-Target Tracking System

A real-time computer vision system simulating a **Ground Control Station (GCS)** dashboard for UAV surveillance. Detects and tracks multiple objects simultaneously using **YOLOv8** and a custom **Centroid Tracker**, displayed inside a **Tkinter desktop dashboard**.

---

## 📸 Demo

> *Add your screen recording or screenshot here*

---

## 🎯 Features

- 🟢 **Real-time object detection** using YOLOv8n (people, vehicles, and 80+ object classes)
- 🟡 **Multi-object tracking** with persistent IDs across frames using custom Centroid Tracker
- 🔵 **Motion trail visualization** — color trails showing movement history per tracked object
- 📊 **Live GCS dashboard** — FPS, detection count, active track count updated in real time
- ⏸ **Pause / Stop controls** — fully functional buttons for session control
- ⚡ **Frame skip optimization** — processes every other frame to maximize FPS on CPU

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| OpenCV | Video capture, frame processing, drawing |
| YOLOv8 (Ultralytics) | Deep learning object detection |
| NumPy | Distance matrix computation for tracking |
| Tkinter | GCS desktop GUI |
| Pillow | Frame rendering inside Tkinter canvas |
| Threading | Async video loop for non-blocking UI |

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/uav-tracker
cd uav-tracker

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python stage6.py
```

> YOLOv8n weights (~6MB) are downloaded automatically on first run.

---

## 🚀 How to Use

1. Run the app — GCS dashboard opens
2. Wait ~3 seconds for YOLOv8 to load
3. Move in front of webcam — detection and tracking begins
4. Watch **motion trails** follow your movement
5. Monitor live **FPS, Detections, Tracks** in the right panel
6. Press **⏸ Pause** to freeze feed, **⏹ Stop** to exit cleanly

---

## 🏗 Architecture

```
Webcam Feed
    ↓
OpenCV Frame Capture (threaded)
    ↓
YOLOv8n Detection → [x1,y1,x2,y2, label, confidence]
    ↓
Confidence Filter (> 0.45)
    ↓
Centroid Tracker → Euclidean distance matching → Persistent IDs
    ↓
OpenCV Drawing → Bounding boxes + Labels + Trails + IDs
    ↓
Tkinter Canvas → Live GCS Dashboard
```

---

## 💡 Key Engineering Decisions

- **Frame skipping** — YOLO runs on every other frame, doubling effective FPS on CPU without losing tracking continuity
- **Centroid matching** — objects matched by minimum Euclidean distance (threshold: 120px), balancing speed vs accuracy
- **Trail history** — last 20 centroids stored per ID, giving smooth motion visualization without memory overhead
- **Lazy model loading** — YOLOv8 loads on first frame so the GUI opens instantly without blocking

---

## 📁 Project Structure

```
uav-tracker/
├── stage6.py          ← Main application
├── requirements.txt   ← Dependencies
└── README.md
```

---

## 📦 Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=9.0.0
numpy>=1.24.0
```
