from flask import Flask, render_template, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)

# --- Load Model and Setup MediaPipe ---
try:
    with open('model.p', 'rb') as f:
        model = pickle.load(f)['model']
except FileNotFoundError:
    print("FATAL ERROR: model.p not found.")
    exit()

mp_hands = mp.solutions.hands
### NEW: Make MediaPipe less picky to improve detection on compressed video ###
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
# ------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    ### NEW: Add a log to show that the server received a request ###
    print("Received a frame for processing...")
    
    json_data = request.get_json()
    image_data = json_data['image_data'].split(',')[1]
    
    decoded_image = base64.b64decode(image_data)
    np_arr = np.frombuffer(decoded_image, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    predicted_character = ""
    landmarks_for_json = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        ### NEW: Add a log to show that a hand was FOUND ###
        print("SUCCESS: Hand detected in frame!")
        
        data_aux, x_, y_ = [], [], []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                landmark = hand_landmarks.landmark[i]
                landmarks_for_json.append({'x': landmark.x, 'y': landmark.y})
                x_.append(landmark.x)
                y_.append(landmark.y)

            min_x, min_y = min(x_), min(y_)
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min_x)
                data_aux.append(hand_landmarks.landmark[i].y - min_y)
        
        if data_aux:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]
            ### NEW: Add a log to show the prediction ###
            print(f"Prediction result: {predicted_character}")

    else:
        ### NEW: Add a log for when NO hand is found ###
        print("INFO: No hand detected in this frame.")

    return jsonify({'prediction': predicted_character, 'landmarks': landmarks_for_json})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
