import numpy as np
import os
file_path=os.path.join('dataset', 'Hello', '0.npy') 

if os.path.exists(file_path):
    data=np.load(file_path)

    frames=data.shape[0]
    total_landmarks=data.shape[1]
    print(f"Frames: {frames} (Should be 30)")
    print(f"Total Values per Frame: {total_landmarks} (Should be 1662)")

    pose_len=33*4    # 132
    face_len=468*3   # 1404
    lh_len=21*3    # 63
    rh_len=21*3    # 63
    expected_total=pose_len+face_len+lh_len+rh_len
    
    if total_landmarks==expected_total:
        print(f"\nMATCH! {pose_len} (Pose) + {face_len} (Face) + {lh_len} (LH) + {rh_len} (RH) = {total_landmarks}")
        print("Your dataset contains ALL landmarks.")
    else:
        print(f"\nMISMATCH! Expected {expected_total}, but got {total_landmarks}.")
else:
    print("File not found. Record some data first!")