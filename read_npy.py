import numpy as np
import os

# Change this to the path of one of your files
file_path = os.path.join('dataset', 'Hello', '0.npy') 

if os.path.exists(file_path):
    data = np.load(file_path)
    
    print(f"✅ File Loaded Successfully: {file_path}")
    print(f"📏 Shape: {data.shape}  (Should be 30 frames, 1662 landmarks)")
    print(f"🔢 Data Type: {data.dtype}")
    
    print("\n--- First Frame, First 5 Landmarks ---")
    print(data[0][:5]) # Prints the first 5 numbers of the first frame
else:
    print("❌ File not found. Check the path!")