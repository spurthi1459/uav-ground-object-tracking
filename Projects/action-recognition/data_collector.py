import argparse
import os
import time

import cv2
import numpy as np

from utils.pose_extractor import PoseExtractor

#CONFIG
SEQUENCE_LENGTH = 30        # frames per sample
DATA_DIR = "data/raw"
ACTIONS = ["walking", "jumping", "sitting", "standing", "waving"]



def collect(action: str, num_samples: int):
    os.makedirs(os.path.join(DATA_DIR, action), exist_ok=True)

    extractor = PoseExtractor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    print(f"\n[INFO] Collecting data for action: '{action}'")
    print(f"[INFO] Target samples: {num_samples}  |  Sequence length: {SEQUENCE_LENGTH} frames")
    print("[INFO] Press SPACE to start recording, Q to quit.\n")

    collected = 0
    recording = False
    buffer: list[np.ndarray] = []

    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        keypoints, annotated, detected = extractor.extract(frame)

        # HUD
        status = "RECORDING" if recording else "PAUSED  (press SPACE)"
        color = (0, 0, 255) if recording else (200, 200, 0)
        cv2.putText(annotated, f"Action: {action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"Collected: {collected}/{num_samples}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if recording:
            cv2.putText(annotated, f"Buffer: {len(buffer)}/{SEQUENCE_LENGTH}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Data Collector", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            recording = not recording
            buffer = []

        if recording and detected:
            buffer.append(keypoints)
            if len(buffer) == SEQUENCE_LENGTH:
                seq = np.array(buffer)          # (30, 132)
                save_path = os.path.join(DATA_DIR, action, f"{collected:04d}.npy")
                np.save(save_path, seq)
                collected += 1
                buffer = []
                print(f"  Saved sample {collected}/{num_samples}", end="\r")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"\n[DONE] Saved {collected} samples to {DATA_DIR}/{action}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect pose sequences for action recognition.")
    parser.add_argument("--action", required=True, choices=ACTIONS, help="Action class to record")
    parser.add_argument("--samples", type=int, default=200, help="Number of sequences to collect")
    args = parser.parse_args()

    collect(args.action, args.samples)
