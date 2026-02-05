import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import mediapipe as mp

# -----------------------------------------------------------
# 1. SETUP
# -----------------------------------------------------------
# Load your trained model
model = load_model('action.keras')

# Get the list of words (Must match Training order!)
# We read the folder names again to ensure exact order
DATA_PATH = os.path.join('dataset')
actions = np.array(os.listdir(DATA_PATH))
print(f"Loaded labels: {actions}")

# Setup MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Variables
sequence = []
sentence = []
threshold = 0.7 # Only show if confidence is above 70%

# -----------------------------------------------------------
# 2. KEYPOINT EXTRACTION FUNCTION (Must match training!)
# -----------------------------------------------------------
def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
        
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468*3)
    
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
        
    return np.concatenate([pose, face, lh, rh])

# -----------------------------------------------------------
# 3. MAIN LOOP
# -----------------------------------------------------------
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Process Frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw Landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 1. Prediction Logic
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:] # Keep last 30 frames
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                # Get best prediction
                best_class_index = np.argmax(res)
                confidence = res[best_class_index]
                prediction = actions[best_class_index]
                
                # Visual Feedback logic
                if confidence > threshold:
                    # Only update if word changed
                    if len(sentence) > 0:
                        if prediction != sentence[-1]:
                            sentence.append(prediction)
                    else:
                        sentence.append(prediction)

                # Keep only last 5 words
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Show Live Probability Bar
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, f"{prediction} ({confidence*100:.1f}%)", (10,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Show "Waiting" if no hands
            cv2.putText(image, "Waiting for hands...", (10,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow('Sign Language Detector', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()