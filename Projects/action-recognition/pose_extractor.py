import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class PoseExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.num_keypoints = 33
        self.feature_size = self.num_keypoints * 4  # x, y, z, visibility

    def extract(self, frame):
       
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        annotated = frame.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            keypoints = self._landmarks_to_array(results.pose_landmarks)
            return keypoints, annotated, True
        else:
            return np.zeros(self.feature_size), annotated, False

    def _landmarks_to_array(self, landmarks):
        
        data = []
        for lm in landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(data, dtype=np.float32)

    def close(self):
        self.pose.close()


def draw_label(frame, label, confidence=None, position=(30, 60)):

    text = label
    if confidence is not None:
        text += f"  {confidence*100:.1f}%"
    cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 2, cv2.LINE_AA
    )
    return frame
