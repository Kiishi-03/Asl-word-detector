from flask import Flask, render_template, Response
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time  # We need the time library for our timer

app = Flask(__name__)

# --- Load the Model and Setup MediaPipe ---
try:
    with open('model.p', 'rb') as f:
        model = pickle.load(f)['model']
except FileNotFoundError:
    print("FATAL ERROR: model.p not found. Make sure it's in the same directory.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
# ---------------------------------------------

# --- NEW: State Variables for Word Building ---
sentence = ""  # The string to hold the typed words
last_stable_prediction = ""  # The last character that was held steadily
prediction_start_time = 0  # The timestamp when the last stable prediction began
HOLD_DURATION = 3.0  # Hold for 3 seconds to type a letter (you can change this to 5.0)


# ---------------------------------------------

def generate_frames():
    # Make our state variables accessible inside this function
    global sentence, last_stable_prediction, prediction_start_time

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Your existing processing logic
            data_aux, x_, y_ = [], [], []
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            predicted_character = ""  # Current frame's prediction

            if results.multi_hand_landmarks:
                # (Your landmark drawing and extraction code is the same)
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)
                    min_x, min_y = min(x_), min(y_)
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min_x)
                        data_aux.append(landmark.y - min_y)

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]

                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

            # --- NEW: Logic for Holding a Sign ---
            current_time = time.time()
            if predicted_character and predicted_character == last_stable_prediction:
                # If the current prediction has been held long enough...
                if current_time - prediction_start_time >= HOLD_DURATION:
                    # Special handling for 'space', 'del', and 'nothing'
                    if predicted_character == 'space':
                        sentence += " "
                    elif predicted_character == 'del':
                        sentence = sentence[:-1]  # Delete last character
                    elif predicted_character != 'nothing':
                        sentence += predicted_character

                    # Reset the timer to prevent immediate re-typing
                    prediction_start_time = float('inf')
            elif predicted_character and predicted_character != 'nothing':
                # If a new, valid sign is detected, start the timer for it.
                last_stable_prediction = predicted_character
                prediction_start_time = current_time
            else:
                # If no hand or 'nothing' is detected, reset.
                last_stable_prediction = ""
                prediction_start_time = float('inf')
            # ------------------------------------

            # --- NEW: Display the Sentence on the Frame ---
            # Draw a semi-transparent background for the text
            cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 0), -1)
            cv2.putText(frame, sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # ---------------------------------------------

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    # Reset the sentence every time the page is loaded
    global sentence, last_stable_prediction, prediction_start_time
    sentence = ""
    last_stable_prediction = ""
    prediction_start_time = float('inf')
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)