import numpy as np
import os

# --- PATH TO ONE OF YOUR FILES ---
# Update this to point to a file you actually recorded
file_path = os.path.join('dataset', 'Hello', '0.npy') 

if os.path.exists(file_path):
    data = np.load(file_path)
    
    # Get the shape
    frames = data.shape[0]
    total_landmarks = data.shape[1]
    
    print(f"✅ File Found!")
    print(f"🎞️  Frames: {frames} (Should be 30)")
    print(f"🔢 Total Values per Frame: {total_landmarks} (Should be 1662)")
    
    # --- THE MATH CHECK ---
    # Let's verify the components
    pose_len = 33 * 4    # 132
    face_len = 468 * 3   # 1404
    lh_len   = 21 * 3    # 63
    rh_len   = 21 * 3    # 63
    expected_total = pose_len + face_len + lh_len + rh_len
    
    if total_landmarks == expected_total:
        print("\nPERFECT MATCH!")
        print(f"The math holds up: {pose_len} (Pose) + {face_len} (Face) + {lh_len} (LH) + {rh_len} (RH) = {total_landmarks}")
        print("Your dataset contains ALL landmarks.")
    else:
        print(f"\nMISMATCH! Expected {expected_total}, but got {total_landmarks}.")
        print("Check your extract_keypoints function.")

else:
    print("File not found. Record some data first!")