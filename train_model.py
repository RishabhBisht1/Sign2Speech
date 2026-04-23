import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# =========================
# LOAD DATA
# =========================
DATA_PATH = os.path.join('datasetV')

actions = np.array(sorted(os.listdir(DATA_PATH)))
print(f"Detected Actions: {actions}")

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Loading data...")

for action in actions:
    action_path = os.path.join(DATA_PATH, action)

    for file_name in os.listdir(action_path):
        if file_name.endswith('.npy'):
            window = np.load(os.path.join(action_path, file_name))
            sequences.append(window)
            labels.append(label_map[action])

print(f"Loaded {len(sequences)} samples.")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,shuffle=True,stratify=y
)

print(f"Training Shape: {X_train.shape}")

# =========================
# MODEL
# =========================
model = Sequential()
model.add(Input(shape=(30, 1662)))

model.add(LSTM(64, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))

model.add(LSTM(32, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# CALLBACKS
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

tensorboard = TensorBoard(log_dir='Logs')

# =========================
# TRAINING
# =========================
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, tensorboard]
)

# =========================
# SAVE MODEL
# =========================
model.save('action.keras')

# =========================
# EVALUATION
# =========================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# =========================
# PREDICTIONS (FIXED PART)
# =========================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

# =========================
# CLASSIFICATION REPORT
# =========================
labels_present = np.unique(y_true)

print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred_classes,
    labels=labels_present,
    target_names=[actions[i] for i in labels_present]
))

# =========================
# CONFUSION MATRIX (OPTIONAL BUT IMPORTANT)
# =========================
import seaborn as sns

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# ACCURACY PLOT
# =========================
plt.figure(figsize=(10, 5))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.show()

print("\nTraining Completed Successfully")