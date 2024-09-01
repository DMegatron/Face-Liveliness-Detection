import cv2
import mediapipe as mp
import numpy as np

class LivelinessDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.base_ear_threshold = 0.585  # Base threshold for EAR
        self.max_tilt = 15  # Maximum tilt in degrees before classifying as fake

    def _calculate_ear(self, landmarks, indices):
        A = np.linalg.norm(np.array(landmarks[indices[1]]) - np.array(landmarks[indices[5]]))
        B = np.linalg.norm(np.array(landmarks[indices[2]]) - np.array(landmarks[indices[4]]))
        C = np.linalg.norm(np.array(landmarks[indices[0]]) - np.array(landmarks[indices[3]]))
        ear = (A + B) / (2.0 * C)
        return ear

    def _calculate_tilt_angle(self, landmarks):
        # Calculate the angle between the eyes to estimate tilt
        left_eye_center = np.array(landmarks[33])
        right_eye_center = np.array(landmarks[263])
        delta_y = right_eye_center[1] - left_eye_center[1]
        delta_x = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return angle

    def detect_liveliness(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

                # Calculate the tilt angle
                tilt_angle = self._calculate_tilt_angle(landmarks)
                print(f"Tilt Angle: {tilt_angle}")

                # Check if the face is tilted beyond the allowed threshold
                if abs(tilt_angle) > self.max_tilt:
                    cv2.putText(frame, 'Fake Face Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Proceed with EAR-based detection if the face is reasonably straight
                    left_eye_indices = [33, 7, 160, 158, 133, 153]  # Left eye landmarks
                    right_eye_indices = [362, 263, 249, 390, 373, 380]  # Right eye landmarks

                    left_ear = self._calculate_ear(landmarks, left_eye_indices)
                    right_ear = self._calculate_ear(landmarks, right_eye_indices)

                    # Check if either EAR is below the threshold
                    if left_ear < self.base_ear_threshold or right_ear < self.base_ear_threshold:
                        cv2.putText(frame, 'Real Face Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'Fake Face Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw landmarks (optional)
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        return frame
