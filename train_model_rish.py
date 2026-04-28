# This is the Builder and Educator. It reads all the .npy files from your dataset, pairs them with the correct word labels, splits the data into training and testing sets, and defines the actual architecture of your Neural Network (the LSTM layers). It trains the network to recognize the patterns and saves the learned knowledge.


import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, 'datasetV'))

actions = np.array(sorted(os.listdir(DATA_PATH)))
print(f"Detected Actions: {actions}")

label_map={label:num for num,label in enumerate(actions)}

sequences,labels=[],[]


# we will ignore facial points
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    file_names = os.listdir(action_path)
    
    for file_name in file_names:
        if file_name.endswith('.npy'):
            # 1. Load the original file (Shape: 30, 1662)
            window = np.load(os.path.join(action_path, file_name))
            
            # 2. Slice out the face data
            # window[:, :132] grabs all 30 frames for the Pose
            # window[:, 1536:] grabs all 30 frames for both Hands
            pose = window[:, :132]
            hands = window[:, 1536:]
            
            # 3. Concatenate to make the new shape (30, 258)
            window_no_face = np.concatenate([pose, hands], axis=1)
            
            # Append to our big list
            sequences.append(window_no_face)
            labels.append(label_map[action])

print(f"Loaded {len(sequences)} videos total.")

X = np.array(sequences)
y = to_categorical(labels).astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print(f"Training Data Shape: {X_train.shape}") # This should now say (..., 30, 258)

# --- UPDATE THE MODEL ARCHITECTURE ---
model = Sequential()

# CHANGE: Update the Input shape from 1662 to 258
model.add(Input(shape=(30, 258)))

# Layer 1: LSTM
model.add(LSTM(64, return_sequences=True, activation='tanh'))

# Layer 2: LSTM (Consolidator)
model.add(LSTM(32, return_sequences=False, activation='tanh'))

# Layer 3: Dense
model.add(Dense(32, activation='relu'))

# Output Layer
model.add(Dense(actions.shape[0], activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=800, callbacks=[TensorBoard(log_dir='Logs')])
model.save('action2.keras')

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")