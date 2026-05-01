# This is your Real-Time Application. It connects to your webcam, extracts live MediaPipe landmarks frame-by-frame, keeps a rolling buffer of the last 30 frames, and feeds them into the trained model to predict the ISL sign in real-time.


import cv2
import numpy as np
import os
import tensorflow as tf
load_model = tf.keras.models.load_model
import mediapipe as mp

model = load_model('action2.keras')

DATA_PATH = os.path.join('datasetV')

if os.path.exists(DATA_PATH):
    actions = np.array(sorted(os.listdir(DATA_PATH)))
else:
    actions = [] 
    
print(f"Loaded labels: {actions}")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

# --- VARIABLES ---
sequence = []
sentence = []      # ADDED THIS BACK
predictions = []
threshold = 0.7 

# Variables for Motion Detection
previous_keypoints = None
motion_threshold = 0.03 # Higher = less sensitive. Lower = more sensitive.
is_recording = False

# --- EXTRACT AND NORMALIZE KEYPOINTS (Shoulder Anchors Included) ---
def extract_keypoints(results):
    # 1. FIND THE ANCHOR POINT (Chest)
    if results.pose_landmarks:
        l_shoulder = results.pose_landmarks.landmark[11]
        r_shoulder = results.pose_landmarks.landmark[12]
        anchor_x = (l_shoulder.x + r_shoulder.x) / 2
        anchor_y = (l_shoulder.y + r_shoulder.y) / 2
    else:
        anchor_x, anchor_y = 0.0, 0.0

    # 2. EXTRACT POSE
    if results.pose_landmarks:
        pose = np.array([[res.x - anchor_x, res.y - anchor_y, res.z, res.visibility] 
                         for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
        
    # (Face is ignored to match the new 258-shape model)

    # 3. EXTRACT LEFT HAND
    if results.left_hand_landmarks:
        lh = np.array([[res.x - anchor_x, res.y - anchor_y, res.z] 
                       for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    # 4. EXTRACT RIGHT HAND
    if results.right_hand_landmarks:
        rh = np.array([[res.x - anchor_x, res.y - anchor_y, res.z] 
                       for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
        
    return np.concatenate([pose, lh, rh])

# --- MAIN CAMERA LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Turn the model on, let me use it inside this indented block of code, and the absolute microsecond I am done or the script crashes, automatically shut it down and release the webcam. If we didn't use this, a crashed script would leave your webcam light on, and we'd have to force-quit Python to use our camera again.

# MediaPipe has separate models for Face Mesh, Pose tracking, and Hand tracking. The Holistic model is a specialized, optimized pipeline that runs all of them simultaneously in a coordinated way so your CPU doesn't melt hehe.
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw Landmarks (Removed the dense face mesh drawing to keep the screen cleaner)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # --- PREDICTION LOGIC ---
        keypoints = extract_keypoints(results)
        
        # 1. Continuous Rolling Buffer: Append new frame, keep only the last 30
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # 2. Predict continuously whenever we have a full 30-frame window
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            best_class_index = np.argmax(res)
            confidence = res[best_class_index]
            prediction = actions[best_class_index]
            
            predictions.append(best_class_index)
            
            # Draw confidence on screen for debugging
            cv2.putText(image, f"Predicting: {prediction} ({confidence*100:.1f}%)", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Check for stability (same prediction for last 5 frames), high confidence, and hand presence
            if len(predictions) >= 5 and len(set(predictions[-5:])) == 1:
                if confidence > 0.85 and (results.left_hand_landmarks or results.right_hand_landmarks):
                    if len(sentence) > 0:
                        if prediction != sentence[-1]:
                            sentence.append(prediction)
                    else:
                        sentence.append(prediction)

            if len(sentence) > 5:
                sentence = sentence[-5:]

        # UI Updates
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        
        display_text = " ".join(sentence)
        cv2.putText(image, display_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Detector', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
