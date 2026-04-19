# This is your Real-Time Application. It connects to your webcam, extracts live MediaPipe landmarks frame-by-frame, keeps a rolling buffer of the last 30 frames, and feeds them into the trained model to predict the ISL sign in real-time.


import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import mediapipe as mp

model = load_model('action.keras')

DATA_PATH = os.path.join('dataset')

if os.path.exists(DATA_PATH):
    actions = np.array(sorted(os.listdir(DATA_PATH)))
else:
    actions = [] 
    
print(f"Loaded labels: {actions}")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

sequence = []
sentence = []
threshold = 0.7 

def extract_keypoints(results):
    # --- 1. FIND THE ANCHOR POINT ---
    if results.pose_landmarks:
        # Get Left Shoulder (11) and Right Shoulder (12)
        l_shoulder = results.pose_landmarks.landmark[11]
        r_shoulder = results.pose_landmarks.landmark[12]
        
        # Calculate the midpoint between shoulders (the chest)
        anchor_x = (l_shoulder.x + r_shoulder.x) / 2
        anchor_y = (l_shoulder.y + r_shoulder.y) / 2
    else:
        # Fallback if no body is detected
        anchor_x, anchor_y = 0.0, 0.0

    # --- 2. EXTRACT AND NORMALIZE POSE ---
    if results.pose_landmarks:
        # Notice we subtract anchor_x and anchor_y from res.x and res.y
        pose = np.array([[res.x - anchor_x, res.y - anchor_y, res.z, res.visibility] 
                         for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
        
    # (Face landmarks are completely ignored here to keep the shape at 258)

    # --- 3. EXTRACT AND NORMALIZE LEFT HAND ---
    if results.left_hand_landmarks:
        lh = np.array([[res.x - anchor_x, res.y - anchor_y, res.z] 
                       for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    # --- 4. EXTRACT AND NORMALIZE RIGHT HAND ---
    if results.right_hand_landmarks:
        rh = np.array([[res.x - anchor_x, res.y - anchor_y, res.z] 
                       for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
        
    # Return the normalized Pose + Hands (132 + 63 + 63 = 258)
    return np.concatenate([pose, lh, rh])

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_TESSELATION, 

            mp_drawing.DrawingSpec(color=(80, 255, 255), thickness=1, circle_radius=1),

            mp_drawing.DrawingSpec(color=(0, 256, 128), thickness=1, circle_radius=1)
        )

        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_CONTOURS, 
            mp_drawing.DrawingSpec(color=(80, 255, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 255, 255), thickness=1, circle_radius=1)
        )

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:] 
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                best_class_index = np.argmax(res)
                confidence = res[best_class_index]
                prediction = actions[best_class_index]
                
                if confidence > threshold:
                    if len(sentence) > 0:
                        if prediction != sentence[-1]:
                            sentence.append(prediction)
                    else:
                        sentence.append(prediction)

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, f"{prediction} ({confidence*100:.1f}%)", (10,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Waiting for hands...", (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow('Sign Language Detector', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
