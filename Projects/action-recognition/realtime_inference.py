import argparse
import os
import time
import json
from collections import deque

import cv2
import numpy as np
import torch

# Direct imports — all files are in the same folder
from pose_extractor import PoseExtractor
from train_model import ActionLSTM

MODEL_PATH        = os.path.join("models", "action_lstm_best.pth")
LABEL_MAP_PATH    = os.path.join("models", "label_map.json")
SEQUENCE_LEN      = 30
INPUT_SIZE        = 132
HIDDEN_SIZE       = 128
NUM_LAYERS        = 2
CONFIDENCE_THRESH = 0.6
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

PALETTE = {
    "walking":  (0, 200, 100),
    "jumping":  (0, 100, 255),
    "sitting":  (255, 150, 0),
    "standing": (100, 255, 255),
    "waving":   (255, 50, 200),
}


def run_inference(source=0, save_path=None):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at '{MODEL_PATH}'")
        print("  Run train_model.py first.\n")
        return
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"[ERROR] Label map not found at '{LABEL_MAP_PATH}'")
        print("  Run train_model.py first.\n")
        return

    with open(LABEL_MAP_PATH) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    num_classes = len(label_map)
    model = ActionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, dropout=0.0)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)

    extractor = PoseExtractor()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    buffer     = deque(maxlen=SEQUENCE_LEN)
    smoothed   = deque(maxlen=5)
    pred_label = "Collecting frames..."
    pred_conf  = 0.0
    frame_count = 0
    fps_timer   = time.time()
    fps_val     = 0.0

    print(f"[INFO] Running  |  Device: {DEVICE}  |  Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if source == 0:
            frame = cv2.flip(frame, 1)

        keypoints, annotated, detected = extractor.extract(frame)
        if detected:
            buffer.append(keypoints)

        if len(buffer) == SEQUENCE_LEN:
            seq    = np.array(buffer, dtype=np.float32)
            tensor = torch.tensor(seq).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1).squeeze()
                conf, idx = probs.max(0)
            conf  = conf.item()
            label = label_map[idx.item()]
            if conf >= CONFIDENCE_THRESH:
                smoothed.append(label)
                pred_label = max(set(smoothed), key=smoothed.count)
                pred_conf  = conf
            else:
                pred_label = "Uncertain"
                pred_conf  = conf

        color = PALETTE.get(pred_label, (200, 200, 200))
        cv2.rectangle(annotated, (10, 10), (430, 85), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (430, 85), color, 2)
        cv2.putText(annotated, pred_label, (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2, cv2.LINE_AA)
        if pred_conf > 0:
            cv2.rectangle(annotated, (10, 88),
                          (10 + int(420 * pred_conf), 98), color, -1)

        frame_count += 1
        if frame_count % 30 == 0:
            fps_val   = 30 / (time.time() - fps_timer)
            fps_timer = time.time()
        cv2.putText(annotated, f"FPS: {fps_val:.1f}",
                    (annotated.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow("Action Recognition  |  Q to quit", annotated)
        if writer:
            writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            os.makedirs("outputs", exist_ok=True)
            sc = f"outputs/screenshot_{int(time.time())}.jpg"
            cv2.imwrite(sc, annotated)
            print(f"Screenshot: {sc}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0)
    parser.add_argument("--save",   default=None)
    args = parser.parse_args()
    src = args.source
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass
    run_inference(source=src, save_path=args.save)
