import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import TensorBoard

DATA_PATH = os.path.join('dataset') 

# Get the list of actions (words) directly from the folder names
# dataset/Hello, dataset/Thanks, etc.
actions = np.array(os.listdir(DATA_PATH))
print(f"Detected Actions: {actions}")

label_map={label:num for num,label in enumerate(actions)}

sequences,labels=[],[]

print("Loading data... (This might take a few seconds)")




for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    file_names = os.listdir(action_path)
    
    for file_name in file_names:
        if file_name.endswith('.npy'):
            # Load file
            window = np.load(os.path.join(action_path, file_name))
            
            # Append to our big list
            sequences.append(window)
            labels.append(label_map[action])

print(f"Loaded {len(sequences)} videos total.")

X = np.array(sequences)
y = to_categorical(labels).astype(int) # Converts [0, 1, 0] -> [1,0,0], [0,1,0]...


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.05)

print(f"Training Data Shape: {X_train.shape}")




model = Sequential()
model.add(Input(shape=(30, 1662)))

# Layer 1: LSTM (Fast learner)
model.add(LSTM(64, return_sequences=True, activation='relu'))

# Layer 2: LSTM (Consolidator)
model.add(LSTM(32, return_sequences=False, activation='relu'))

# Layer 3: Dense (Decision maker)
model.add(Dense(32, activation='relu'))

# Output Layer
model.add(Dense(actions.shape[0], activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=500, callbacks=[TensorBoard(log_dir='Logs')])
model.save('action.keras')

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")