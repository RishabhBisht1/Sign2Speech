import cv2
import numpy as np
import os
import mediapipe as mp

DATA_PATH = os.path.join('dataset') 
SEQUENCE_LENGTH = 30 # Frames per video

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """ Extracts and flattens all landmarks. """
    # 1. Pose
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
    
    # 2. Face
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468*3)
    
    # 3. Left Hand
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    # 4. Right Hand
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
        
    return np.concatenate([pose, face, lh, rh])


action_name = input("Enter the word you want to record (e.g., Hello): ").strip()

# Create folder if it doesn't exist
action_path = os.path.join(DATA_PATH, action_name)
if not os.path.exists(action_path):
    os.makedirs(action_path)

# Check how many videos already exist (so we don't overwrite)
existing_files = os.listdir(action_path)
# We count files that end in .npy to get the next sequence number
start_sequence = len([f for f in existing_files if f.endswith('.npy')])

print(f"--- READY TO RECORD: '{action_name}' ---")
print(f"Existing videos: {start_sequence}")
print("Press 'SPACE' to record a sequence.")
print("Press 'Q' to quit.")

cap = cv2.VideoCapture(0)
sequence_count = start_sequence

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        # Read Feed (Preview Mode)
        ret, frame = cap.read()
        if not ret: break

        # Process MediaPipe (for visualization only during preview)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # GUI Text
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, f"Recording: {action_name} | Count: {sequence_count}", (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)

        # Check for Key Press
        key = cv2.waitKey(10) & 0xFF

        if key == 32: # 32 is Spacebar
            print(f"Recording Sequence {sequence_count}...")
            
            # Storage for this video
            window = []
            
            # Record 30 Frames
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                
                # Process MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw (Optional, but good for feedback)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # GUI: "RECORDING"
                cv2.putText(image, f"CAPTURING FRAME {frame_num}/{SEQUENCE_LENGTH}", (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(1) # Very short wait to update screen

                # Save Keypoints
                keypoints = extract_keypoints(results)
                window.append(keypoints)
            
            # SAVE TO DISK
            npy_path = os.path.join(DATA_PATH, action_name, str(sequence_count))
            np.save(npy_path, np.array(window))
            
            print(f"Saved {action_name}_{sequence_count}.npy")
            sequence_count += 1

        # Quit
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()