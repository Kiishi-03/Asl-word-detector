from flask import Flask, render_template, Response
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import os

app = Flask(__name__)

# --- Load the Model and Setup MediaPipe ---
try:
    with open('model.p', 'rb') as f:
        model = pickle.load(f)['model']
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ FATAL ERROR: model.p not found. Make sure it's in the same directory.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Updated MediaPipe settings for better detection
hands = mp_hands.Hands(
    static_image_mode=False,  # Changed to False for video stream
    max_num_hands=1,
    min_detection_confidence=0.5,  # Increased confidence
    min_tracking_confidence=0.5   # Added tracking confidence
)

# --- State Variables for Word Building ---
sentence = ""
last_stable_prediction = ""
prediction_start_time = 0
HOLD_DURATION = 5.0  # 5 seconds as requested

def generate_frames():
    global sentence, last_stable_prediction, prediction_start_time

    # Try different camera indices if default doesn't work
    cap = None
    for i in range(3):  # Try cameras 0, 1, 2
        try:
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:  # Test if we can read a frame
                print(f"✅ Using camera {i}")
                break
            cap.release()
        except:
            continue
    
    if cap is None:
        print("❌ No camera found!")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_WIDTH, 640)
    cap.set(cv2.CAP_PROP_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read frame")
            break
        
        # ✅ FLIP THE FRAME (This was missing!)
        frame = cv2.flip(frame, 1)
        
        # Process frame
        data_aux, x_, y_ = [], [], []
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand landmarks
        results = hands.process(frame_rgb)
        
        predicted_character = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # ✅ DRAW HAND LANDMARKS AND CONNECTIONS
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmark coordinates
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                
                # Normalize coordinates
                min_x, min_y = min(x_), min(y_)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

            # Make prediction
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]
                
                # Draw bounding box and prediction
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
                
            except Exception as e:
                print(f"Prediction error: {e}")

        # ✅ IMPROVED LOGIC FOR HOLDING A SIGN
        current_time = time.time()
        
        if predicted_character and predicted_character == last_stable_prediction:
            hold_time = current_time - prediction_start_time
            
            # Show progress bar for holding
            if hold_time < HOLD_DURATION:
                progress = int((hold_time / HOLD_DURATION) * 200)  # 200px wide progress bar
                cv2.rectangle(frame, (W//2 - 100, 50), (W//2 + 100, 70), (50, 50, 50), -1)
                cv2.rectangle(frame, (W//2 - 100, 50), (W//2 - 100 + progress, 70), (0, 255, 0), -1)
                cv2.putText(frame, f"Hold: {hold_time:.1f}s", (W//2 - 50, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # If held long enough, add to sentence
            if hold_time >= HOLD_DURATION:
                if predicted_character == 'space':
                    sentence += " "
                elif predicted_character == 'del':
                    sentence = sentence[:-1]
                elif predicted_character != 'nothing':
                    sentence += predicted_character
                
                # Reset timer to prevent immediate re-typing
                prediction_start_time = float('inf')
                last_stable_prediction = ""
                
        elif predicted_character and predicted_character != 'nothing':
            # New valid sign detected, start timer
            last_stable_prediction = predicted_character
            prediction_start_time = current_time
        else:
            # No hand or 'nothing' detected, reset
            last_stable_prediction = ""
            prediction_start_time = float('inf')

        # ✅ DISPLAY SENTENCE WITH BETTER STYLING
        # Create a semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display the sentence
        display_text = sentence if sentence else "Start signing to type..."
        cv2.putText(frame, display_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    # Reset the sentence every time the page is loaded
    global sentence, last_stable_prediction, prediction_start_time
    sentence = ""
    last_stable_prediction = ""
    prediction_start_time = float('inf')
    print("🔄 Page loaded, sentence reset")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear')
def clear_sentence():
    global sentence, last_stable_prediction, prediction_start_time
    sentence = ""
    last_stable_prediction = ""
    prediction_start_time = float('inf')
    return "Sentence cleared"

if __name__ == '__main__':
    # For production on Render, use environment port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
