import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import TensorBoard

# ------------------------------------------------------------------------
# 1. SETUP & DATA LOADING
# ------------------------------------------------------------------------
DATA_PATH = os.path.join('dataset') 

# Get the list of actions (words) directly from the folder names
# This assumes your folder structure is: dataset/Hello, dataset/Thanks, etc.
actions = np.array(os.listdir(DATA_PATH))
print(f"Detected Actions: {actions}")

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Loading data... (This might take a few seconds)")

for action in actions:
    # Get all .npy files for this action
    action_path = os.path.join(DATA_PATH, action)
    file_names = os.listdir(action_path)
    
    for file_name in file_names:
        if file_name.endswith('.npy'):
            # Load the file
            window = np.load(os.path.join(action_path, file_name))
            
            # Append to our big list
            sequences.append(window)
            labels.append(label_map[action])

print(f"✅ Loaded {len(sequences)} videos total.")

# ------------------------------------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------------------------------------
X = np.array(sequences)
y = to_categorical(labels).astype(int) # Converts [0, 1, 0] -> [1,0,0], [0,1,0]...

# Split: 95% Training, 5% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(f"Training Data Shape: {X_train.shape}") # Should be (Videos, 30, 1662)

# ------------------------------------------------------------------------
# 3. BUILD THE LSTM MODEL
# ------------------------------------------------------------------------
model = Sequential()

# Layer 1: LSTM (Input)
# We use 'Input' layer to define shape explicitly. Shape = (30 frames, 1662 landmarks)
model.add(Input(shape=(30, 1662)))
model.add(LSTM(64, return_sequences=True, activation='relu'))

# Layer 2: LSTM
model.add(LSTM(128, return_sequences=True, activation='relu'))

# Layer 3: LSTM (Last LSTM layer, so return_sequences=False)
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Layer 4: Dense (Interpretation)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Layer 5: Output (Prediction)
# actions.shape[0] dynamically sets the output neurons (e.g. 3 words = 3 neurons)
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ------------------------------------------------------------------------
# 4. TRAIN
# ------------------------------------------------------------------------
print("Starting Training...")
# Epochs: How many times to loop through the data. 
# 100-200 is usually good for small datasets.
model.fit(X_train, y_train, epochs=200, callbacks=[TensorBoard(log_dir='Logs')])

# ------------------------------------------------------------------------
# 5. SAVE
# ------------------------------------------------------------------------
model.save('action.keras') # Saves the trained brain
print("✅ Model Saved as 'action.keras'")

# (Optional) Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")