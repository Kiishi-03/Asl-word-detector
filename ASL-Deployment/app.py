from flask import Flask, render_template, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64 # To decode images sent from the browser

app = Flask(__name__)

# --- Load Model and Setup MediaPipe (This part is the same) ---
try:
    with open('model.p', 'rb') as f:
        model = pickle.load(f)['model']
except FileNotFoundError:
    print("FATAL ERROR: model.p not found.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
# -------------------------------------------------------------

# This route serves our main HTML page.
@app.route('/')
def index():
    return render_template('index.html')

# This is our new API endpoint. It only accepts POST requests.
@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the JSON data sent from the browser
    json_data = request.get_json()
    # Extract the image data, which is a Base64 encoded string
    image_data = json_data['image_data'].split(',')[1]
    
    # Decode the Base64 string into bytes, then into a NumPy array, and finally into an OpenCV image
    decoded_image = base64.b64decode(image_data)
    np_arr = np.frombuffer(decoded_image, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Initialize variables for this frame
    predicted_character = ""
    landmarks_for_json = []  # This list will hold landmark coordinates to send back

    # Convert the image to RGB and process with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # If a hand is found...
    if results.multi_hand_landmarks:
        data_aux, x_, y_ = [], [], []
        for hand_landmarks in results.multi_hand_landmarks:
            # --- POPULATE THE LANDMARK LIST FOR THE FRONTEND ---
            # Loop through all 21 landmarks
            for i in range(len(hand_landmarks.landmark)):
                landmark = hand_landmarks.landmark[i]
                # Add the x and y coordinates to our list
                landmarks_for_json.append({'x': landmark.x, 'y': landmark.y})
                x_.append(landmark.x)
                y_.append(landmark.y)
            
            # --- PREPARE DATA FOR THE MODEL (This logic is the same as before) ---
            min_x, min_y = min(x_), min(y_)
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min_x)
                data_aux.append(hand_landmarks.landmark[i].y - min_y)
        
        # Only predict if we actually have data
        if data_aux:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

    # --- RETURN THE COMPLETE DATA PACKET ---
    # We send back a JSON object with two keys: the prediction and the list of landmarks.
    return jsonify({'prediction': predicted_character, 'landmarks': landmarks_for_json})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
