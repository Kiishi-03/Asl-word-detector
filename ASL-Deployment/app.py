from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import pickle
import mediapipe as mp
import json
from collections import Counter
import time

app = Flask(__name__)

# Load the trained model
try:
    with open('model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Global variables for prediction smoothing
prediction_buffer = []
buffer_size = 10
last_prediction = ""
prediction_confidence = 0.0

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        global prediction_buffer, last_prediction, prediction_confidence
        
        success, frame = self.video.read()
        if not success:
            return None
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        prediction_text = "No hand detected"
        confidence = 0.0
        
        if results.multi_hand_landmarks and model:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmarks for prediction
                data_aux = []
                x_ = []
                y_ = []
                
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                
                # Normalize landmarks relative to bounding box
                if x_ and y_:
                    min_x, min_y = min(x_), min(y_)
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min_x)
                        data_aux.append(lm.y - min_y)
                    
                    # Make prediction
                    try:
                        # Ensure consistent feature length (42 features for 21 landmarks)
                        if len(data_aux) == 42:
                            prediction = model.predict([np.asarray(data_aux)])
                            predicted_character = str(prediction[0])
                            
                            # Get prediction probability if available
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba([np.asarray(data_aux)])
                                confidence = np.max(proba)
                            else:
                                confidence = 0.8  # Default confidence
                            
                            # Add to buffer for smoothing
                            prediction_buffer.append(predicted_character)
                            if len(prediction_buffer) > buffer_size:
                                prediction_buffer.pop(0)
                            
                            # Get most common prediction from buffer
                            if prediction_buffer:
                                most_common = Counter(prediction_buffer).most_common(1)[0]
                                prediction_text = most_common[0]
                                prediction_confidence = confidence
                            
                    except Exception as e:
                        prediction_text = f"Prediction error: {str(e)}"
                        print(f"Prediction error: {e}")
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Background rectangle for text
        text_size = cv2.getTextSize(prediction_text, font, font_scale, thickness)[0]
        cv2.rectangle(frame, (10, 10), (text_size[0] + 20, text_size[1] + 30), (0, 0, 0), -1)
        
        # Prediction text
        cv2.putText(frame, prediction_text, (20, 40), font, font_scale, (0, 255, 0), thickness)
        
        # Confidence text
        conf_text = f"Confidence: {prediction_confidence:.2f}"
        cv2.putText(frame, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        instruction = "Show your hand to the camera for ASL recognition"
        cv2.putText(frame, instruction, (20, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        last_prediction = prediction_text
        return frame

def generate_frames():
    camera = VideoCamera()
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({
        'prediction': last_prediction,
        'confidence': prediction_confidence
    })

if __name__ == '__main__':
    import os
    print("Starting ASL Recognition Web App...")
    print("Make sure your model.p file is in the same directory!")
    
    # Use Render's PORT environment variable, fallback to 5000 for local development
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('RENDER') != 'true'  # Disable debug in production
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
