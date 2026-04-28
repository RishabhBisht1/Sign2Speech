# Data Generator. It turns on the webcam, runs MediaPipe Holistic to find the body/hand landmarks, 
# and records exactly 30 frames of motion when you press the spacebar. 
# It then flattens those points and saves them into the datasetV folder as .npy files.

import cv2
import numpy as np
import os
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, 'datasetV'))
SEQUENCE_LENGTH=30 # Frames per video
# Sentence Construction and Deduplication
mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils


# we use shoulders as anchor point so even if the signs are made on sides of camera it still works
def extract_keypoints(results):
    # --- 1. FIND THE ANCHOR POINT (Chest) ---
    if results.pose_landmarks:
        l_shoulder = results.pose_landmarks.landmark[11]
        r_shoulder = results.pose_landmarks.landmark[12]
        anchor_x = (l_shoulder.x + r_shoulder.x) / 2
        anchor_y = (l_shoulder.y + r_shoulder.y) / 2
    else:
        anchor_x, anchor_y = 0.0, 0.0

    # --- 2. EXTRACT POSE ---
    if results.pose_landmarks:
        pose = np.array([[res.x - anchor_x, res.y - anchor_y, res.z, res.visibility] 
                         for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
        
    # --- 3. EXTRACT FACE (WE KEEP THIS FOR THE DATASET) ---
    if results.face_landmarks:
        face = np.array([[res.x - anchor_x, res.y - anchor_y, res.z] 
                         for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468*3)

    # --- 4. EXTRACT LEFT HAND ---
    if results.left_hand_landmarks:
        lh = np.array([[res.x - anchor_x, res.y - anchor_y, res.z] 
                       for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    # --- 5. EXTRACT RIGHT HAND ---
    if results.right_hand_landmarks:
        rh = np.array([[res.x - anchor_x, res.y - anchor_y, res.z] 
                       for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
        
    # Return Pose + Face + Hands (132 + 1404 + 63 + 63 = 1662)
    return np.concatenate([pose, face, lh, rh])


action_name=input("Enter the word you want to record: ").strip()

action_path=os.path.join(DATA_PATH, action_name)
if not os.path.exists(action_path):
    os.makedirs(action_path)

existing_files=os.listdir(action_path)

start_sequence=len([f for f in existing_files if f.endswith('.npy')])

print("Press 'SPACE' to record a sequence.")
print("Press 'Q' to quit.")

cap=cv2.VideoCapture(0)
sequence_count=start_sequence

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():

        ret,frame=cap.read()
        if not ret:break

        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results=holistic.process(image)
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)

        cv2.rectangle(image,(0,0),(640, 40),(245,117,16),-1)
        cv2.putText(image, f"Recording: {action_name} | Count: {sequence_count}", (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed',image)


        key=cv2.waitKey(10) & 0xFF

        if key==32: # 32 is Spacebar
            print(f"Recording Sequence {sequence_count}...")

            for i in range(40): 
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                countdown_text = "GET READY..." if i < 20 else "GO!"
                cv2.putText(image, countdown_text, (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(1)

            window=[]
            
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                cv2.putText(image, f"CAPTURING FRAME {frame_num}/{SEQUENCE_LENGTH}", (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(1)

                keypoints = extract_keypoints(results)
                window.append(keypoints)
            

            npy_path = os.path.join(DATA_PATH, action_name, str(sequence_count))
            np.save(npy_path, np.array(window))
            
            print(f"Saved {action_name}_{sequence_count}.npy")
            sequence_count += 1

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
