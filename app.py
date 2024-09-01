from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

# Initialize MediaPipe Hands and Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Initialize MediaPipe Hands and Face Mesh
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Global variables
finger_count_global = 0
fingers_followed_by_zero = None
last_finger_count = None
last_zero_time = None
last_finger_time = None
steady_hand_start = None
steady_hand_count = None

def is_point_outside_triangle(pt, v1, v2, v3):
    pt = np.array([pt.x, pt.y])
    v1 = np.array([v1.x, v1.y])
    v2 = np.array([v2.x, v2.y])
    v3 = np.array([v3.x, v3.y])
    centroid = (v1 + v2 + v3) / 3
    scale_factor = 3
    v1 = centroid + scale_factor * (v1 - centroid)
    v2 = centroid + scale_factor * (v2 - centroid)
    v3 = centroid + scale_factor * (v3 - centroid)
    
    d1 = np.sign(np.cross(v2 - v1, pt - v1))
    d2 = np.sign(np.cross(v3 - v2, pt - v2))
    d3 = np.sign(np.cross(v1 - v3, pt - v3))
    
    return not (d1 == d2 == d3)

def count_fingers(landmarks):
    """Count the number of fingers extended based on hand landmarks."""
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_base = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_base = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    def is_finger_extended(tip, pip):
        return tip.y < pip.y
    
    finger_count = 0
    
    if is_point_outside_triangle(thumb_tip, wrist, index_base, pinky_base):
        if is_finger_extended(thumb_tip, thumb_ip):
            finger_count += 1
    
    if is_finger_extended(index_tip, index_pip):
        finger_count += 1
    if is_finger_extended(middle_tip, middle_pip):
        finger_count += 1
    if is_finger_extended(ring_tip, ring_pip):
        finger_count += 1
    if is_finger_extended(pinky_tip, pinky_pip):
        finger_count += 1

    return finger_count

def _calculate_ear(landmarks, eye_indices):
    """Calculate the Eye Aspect Ratio (EAR) to assess if eyes are closed or open."""
    # Eye landmarks
    eye = np.array([landmarks[i] for i in eye_indices])

    # Calculate the distances between the vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Calculate the distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])

    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

def detect_liveliness(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            left_eye_center = np.array(landmarks[33])
            right_eye_center = np.array(landmarks[263])
            delta_y = right_eye_center[1] - left_eye_center[1]
            delta_x = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(delta_y, delta_x))

            if abs(angle) > 20 and abs(delta_y) < 10:
                cv2.putText(frame, 'Fake Face Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                left_eye_indices = [33, 7, 160, 158, 133, 153]
                right_eye_indices = [362, 263, 249, 390, 373, 380]

                left_ear = _calculate_ear(landmarks, left_eye_indices)
                right_ear = _calculate_ear(landmarks, right_eye_indices)

                base_ear_threshold = 0.585
                if left_ear < base_ear_threshold or right_ear < base_ear_threshold:
                    cv2.putText(frame, 'Real Face Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Fake Face Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw the face mesh with reduced density
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                if idx % 1 == 0:  # Reduce density
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    return frame

def generate_frames():
    global finger_count_global, fingers_followed_by_zero, last_finger_count, last_zero_time, last_finger_time, steady_hand_start, steady_hand_count
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, image = cap.read()
        if not success:
            print("Failed to capture image")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

                finger_count = count_fingers(hand_landmarks)
                finger_count_global = finger_count

                # Check hand steady state
                if steady_hand_start is not None:
                    if finger_count == steady_hand_count:
                        if current_time - steady_hand_start >= 2:
                            if last_finger_count != finger_count:
                                last_finger_count = finger_count
                            if last_zero_time and current_time - last_zero_time >= 2:
                                fingers_followed_by_zero = last_finger_count
                            steady_hand_start = current_time
                    else:
                        steady_hand_start = None
                        steady_hand_count = None

                if finger_count == 0:
                    if last_finger_count is not None:
                        last_zero_time = current_time
                        steady_hand_start = current_time
                else:
                    if steady_hand_start is None:
                        steady_hand_start = current_time
                        steady_hand_count = finger_count

                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image.shape[1])
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0]) - 20
                cv2.putText(image, f"Fingers: {finger_count}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Detect face liveliness
        image = detect_liveliness(image)

        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            print("Failed to encode image.")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    print("Released the video capture object.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/finger_count')
def finger_count():
    global finger_count_global
    return jsonify({'finger_count': finger_count_global})

@app.route('/fingers_followed_by_zero')
def fingers_followed_by_zero_route():
    global fingers_followed_by_zero
    return jsonify({'fingers_followed_by_zero': fingers_followed_by_zero})

@app.route('/face_liveliness')
def face_liveliness():
    return jsonify({'status': 'face liveliness endpoint is active'})

if __name__ == '__main__':
    app.run(debug=True)
