import cv2
import numpy as np
import os
import time
import threading
import queue
from tensorflow.keras.models import load_model
import mediapipe as mp
import google.generativeai as genai
import pyttsx3
import pythoncom
import win32com.client

# 1. API & NLP SETUP
# Replace with your actual Gemini API Key
genai.configure(api_key="API_KEY")

system_instruction = """
You are a highly efficient Indian Sign Language (ISL) to English translator. 
I will give you a sequence of raw translated words (glosses). 
Your job is to convert them into a single, grammatically correct, natural English sentence.
Do not add any conversational filler. Only output the final sentence.
Example Input: "WHERE HOSPITAL" -> Output: "Where is the hospital?"
Example Input: "YOU NAME WHAT" -> Output: "What is your name?"
"""

# Initialize Gemini 1.5 Flash (Fastest model for text)
nlp_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=system_instruction
)

# --- DEDICATED SPEAKER THREAD ---
speech_queue = queue.Queue()

def tts_worker():
    """This function runs forever in the background, natively speaking text"""
    # Register thread with Windows
    pythoncom.CoInitialize()
    
    # Initialize the Native Windows Speaker (Bypassing pyttsx3)
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    
    # Optional: Set the speed (Range is usually -10 to 10, 0 is normal)
    speaker.Rate = 1 
    
    while True:
        text = speech_queue.get() 
        if text is None: 
            break
        
        # Native speak command (this does not deadlock like runAndWait)
        speaker.Speak(text) 
        speech_queue.task_done()

# Start the speaker thread ONCE when the program loads
threading.Thread(target=tts_worker, daemon=True).start()

# 2. MACHINE LEARNING SETUP
# Load the face-free LSTM model
model = load_model('action2.keras') 

DATA_PATH = os.path.join('datasetV')
actions = np.array(sorted(os.listdir(DATA_PATH))) if os.path.exists(DATA_PATH) else []
print(f"Loaded labels: {actions}")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

# Variables for LSTM & Motion Detection
sequence = []
sentence = []      
threshold = 0.7 
previous_keypoints = None
motion_threshold = 0.03 
is_recording = False

# Variables for NLP Timeout
last_sign_time = time.time()
silence_timeout = 2.0 # 2 Seconds of silence triggers translation
final_translation = "Waiting for signs..."
is_translating = False # Prevents spamming the API

# 3. HELPER FUNCTIONS
def extract_keypoints(results):
    # Anchor to the Chest
    if results.pose_landmarks:
        l_shoulder = results.pose_landmarks.landmark[11]
        r_shoulder = results.pose_landmarks.landmark[12]
        anchor_x = (l_shoulder.x + r_shoulder.x) / 2
        anchor_y = (l_shoulder.y + r_shoulder.y) / 2
    else:
        anchor_x, anchor_y = 0.0, 0.0

    # Extract Pose (Normalized)
    if results.pose_landmarks:
        pose = np.array([[res.x - anchor_x, res.y - anchor_y, res.z, res.visibility] 
                         for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
        
    # (Face is ignored, keeping shape at 258 for action2.keras)

    # Extract Hands (Normalized)
    lh = np.array([[res.x - anchor_x, res.y - anchor_y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x - anchor_x, res.y - anchor_y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
    return np.concatenate([pose, lh, rh])

def process_translation(raw_words):
    """Runs in a background thread to prevent webcam freezing"""
    global final_translation, is_translating
    is_translating = True
    
    try:
        raw_string = " ".join(raw_words)
        print(f"\n[API REQUEST] Sending: {raw_string}")
        
        # Call Gemini API
        response = nlp_model.generate_content(raw_string)
        final_translation = response.text.strip()
        
        print(f"[API SUCCESS] Translated: {final_translation}\n")
        
        # PUT THE TEXT IN THE QUEUE (Do not use engine.say here anymore!)
        speech_queue.put(final_translation)
        
    except Exception as e:
        print(f"[API ERROR] {e}")
        final_translation = "Error translating sentence."
        
    is_translating = False

# 4. MAIN CAMERA LOOP
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # --- PREDICTION LOGIC ---
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            
            # Motion Detection Trigger
            if previous_keypoints is not None:
                movement = np.mean(np.abs(keypoints - previous_keypoints))
                if not is_recording and movement > motion_threshold:
                    is_recording = True
                    sequence = [] 
            
            previous_keypoints = keypoints 

            # Sequence Building
            if is_recording:
                sequence.append(keypoints)
                cv2.putText(image, f"Recording Sign... {len(sequence)}/30", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Predict at 30 frames
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    best_class_index = np.argmax(res)
                    confidence = res[best_class_index]
                    prediction = actions[best_class_index]
                    
                    if confidence > threshold:
                        # Append word if it's new, and RESET the 2-second timer
                        if len(sentence) == 0 or prediction != sentence[-1]:
                            sentence.append(prediction)
                            last_sign_time = time.time() 
                            print(f"Detected: {prediction}")

                    is_recording = False
                    sequence = [] 
        else:
            previous_keypoints = None
            is_recording = False
            sequence = []

        # --- NLP TRANSLATION TRIGGER ---
        # If we have words, 2 seconds have passed since the last sign, and we aren't currently translating
        if len(sentence) > 0 and (time.time() - last_sign_time) > silence_timeout and not is_translating:
            final_translation = "Translating..."
            
            # Copy the sentence buffer and clear the main one so the user can start signing the next sentence immediately
            words_to_translate = sentence.copy()
            sentence = []
            
            # Start the API call in a background thread
            threading.Thread(target=process_translation, args=(words_to_translate,)).start()

        # --- UI UPDATES ---
        # Draw top black bar for text
        cv2.rectangle(image, (0,0), (1280, 80), (0, 0, 0), -1)
        
        # Display Raw Buffer
        raw_text = "Raw ISL: " + " ".join(sentence)
        cv2.putText(image, raw_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Display Final English Translation
        cv2.putText(image, final_translation, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Sign Language Translator', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()