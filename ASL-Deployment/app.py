from flask import Flask, request, jsonify, render_template_string
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image
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
mp_drawing = mp.solutions.drawing_utils # To draw landmarks
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_image(image_data):
    """Process base64 image, draw landmarks, and return prediction + annotated image"""
    prediction_result = "nothing"
    
    try:
        # Decode the image and prepare the frame
        image_data_url = image_data
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # This will be the image we send back. Start with the original.
        annotated_frame = frame.copy()

        # Process with MediaPipe
        data_aux, x_, y_ = [], [], []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks ON TOP of our copied frame
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                
                min_x, min_y = min(x_), min(y_)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)
            
            prediction = model.predict([np.asarray(data_aux)])
            prediction_result = prediction[0]
        
        # Encode the annotated frame (with or without new landmarks) to send back
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_image_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        return prediction_result, annotated_image_b64
        
    except Exception as e:
        print(f"Error processing image: {e}")
        # On error, return the original image data URL
        return "error", image_data_url

@app.route('/')
def index():
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ASL to Text Translator</title>
        <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { margin: 0; padding: 0; }
        </style>
    </head>
    <body>
        <div id="root"></div>
        
        <script type="text/babel">
            const { useState, useRef, useEffect } = React;
            
            const ASLTranslator = () => {
                const videoRef = useRef(null);
                const [sentence, setSentence] = useState('');
                const [currentSign, setCurrentSign] = useState('');
                const [holdProgress, setHoldProgress] = useState(0);
                const [isStreaming, setIsStreaming] = useState(false);
                const [error, setError] = useState('');
                const [lastStablePrediction, setLastStablePrediction] = useState('');
                const [predictionStartTime, setPredictionStartTime] = useState(0);
                
                // ### CHANGE: State to hold the annotated image which will be our main display
                const [displayImage, setDisplayImage] = useState(null);

                const HOLD_DURATION = 5000;
                
                useEffect(() => {
                    startCamera();
                    return () => stopCamera();
                }, []);
                
                const startCamera = async () => {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({
                            video: { width: 640, height: 480, facingMode: 'user' }
                        });
                        
                        if (videoRef.current) {
                            videoRef.current.srcObject = stream;
                            videoRef.current.onloadedmetadata = () => {
                                videoRef.current.play();
                                setIsStreaming(true);
                                setError('');
                                processVideo();
                            };
                        }
                    } catch (err) {
                        setError('Camera access denied. Please allow camera permissions.');
                        console.error('Camera error:', err);
                    }
                };
                
                const stopCamera = () => {
                    if (videoRef.current && videoRef.current.srcObject) {
                        const tracks = videoRef.current.srcObject.getTracks();
                        tracks.forEach(track => track.stop());
                    }
                };
                
                const processVideo = () => {
                    const captureFrame = async () => {
                        if (videoRef.current && videoRef.current.videoWidth > 0) {
                            const canvas = document.createElement('canvas');
                            const context = canvas.getContext('2d');
                            const video = videoRef.current;
                            
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
                            context.drawImage(video, 0, 0);
                            
                            const imageData = canvas.toDataURL('image/jpeg', 0.8);
                            
                            try {
                                const response = await fetch('/predict', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ image: imageData })
                                });
                                
                                if (response.ok) {
                                    const result = await response.json();
                                    // ### CHANGE: Update the source for our main display image
                                    setDisplayImage(result.image);
                                    handlePrediction(result.prediction);
                                }
                            } catch (err) {
                                console.error('Prediction error:', err);
                            }
                            
                            requestAnimationFrame(captureFrame); // Use requestAnimationFrame for smoother loops
                        } else {
                            // If video isn't ready, try again shortly
                            setTimeout(captureFrame, 100);
                        }
                    };
                    
                    captureFrame();
                };
                
                const handlePrediction = (prediction) => {
                    if (!prediction || prediction === 'nothing') {
                        setCurrentSign('');
                        setLastStablePrediction('');
                        setHoldProgress(0);
                        return;
                    }
                    
                    setCurrentSign(prediction);
                    const currentTime = Date.now();
                    
                    if (prediction === lastStablePrediction) {
                        const holdTime = currentTime - predictionStartTime;
                        const progress = Math.min((holdTime / HOLD_DURATION) * 100, 100);
                        setHoldProgress(progress);
                        
                        if (holdTime >= HOLD_DURATION) {
                            if (prediction === 'space') {
                                setSentence(prev => prev + ' ');
                            } else if (prediction === 'del') {
                                setSentence(prev => prev.slice(0, -1));
                            } else {
                                setSentence(prev => prev + prediction);
                            }
                            
                            setLastStablePrediction('');
                            setPredictionStartTime(0);
                            setHoldProgress(0);
                        }
                    } else {
                        setLastStablePrediction(prediction);
                        setPredictionStartTime(currentTime);
                        setHoldProgress(0);
                    }
                };
                
                return React.createElement('div', {
                    className: 'min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center p-4'
                }, React.createElement('div', {
                    className: 'bg-white bg-opacity-95 rounded-2xl p-8 shadow-2xl max-w-4xl w-full'
                }, [
                    React.createElement('h1', { key: 'title', className: 'text-4xl font-bold text-center text-gray-800 mb-6' }, '🤟 ASL to Text Translator'),
                    error && React.createElement('div', { key: 'error', className: 'bg-red-50 border-2 border-red-200 rounded-lg p-4 mb-6' }, React.createElement('span', { className: 'text-red-700 font-medium' }, error)),
                    React.createElement('div', { key: 'sentence-display', className: 'bg-gray-900 rounded-lg p-4 mb-4' }, React.createElement('div', { className: 'text-white text-xl font-mono min-h-8' }, sentence || "Start signing to type...")),
                    
                    React.createElement('div', {
                        key: 'video-container',
                        className: 'relative inline-block rounded-lg overflow-hidden shadow-xl mb-6 bg-black w-[640px] h-[480px] flex items-center justify-center'
                    }, [
                        // ### CHANGE: This video is now hidden. It's just a data source.
                        React.createElement('video', {
                            key: 'video',
                            ref: videoRef,
                            style: { display: 'none' },
                            muted: true,
                            playsInline: true
                        }),
                        
                        // ### CHANGE: This image is now the main display. It shows the processed feed.
                        displayImage ? 
                            React.createElement('img', {
                                key: 'display',
                                src: displayImage,
                                className: 'block w-full h-full',
                                style: { transform: 'scaleX(-1)', objectFit: 'cover' }
                            }) :
                            React.createElement('div', { key: 'placeholder', className: 'text-white' }, 'Starting camera...'),

                        // This overlay for the prediction text remains the same
                        isStreaming && currentSign && React.createElement('div', {
                            key: 'overlay',
                            className: 'absolute top-4 left-4 bg-black bg-opacity-70 text-white px-3 py-2 rounded-lg'
                        }, [
                            React.createElement('div', { key: 'sign', className: 'text-lg font-bold' }, currentSign),
                            holdProgress > 0 && React.createElement('div', { key: 'progress-container', className: 'mt-2' }, [
                                React.createElement('div', { key: 'progress-bg', className: 'bg-gray-600 rounded-full h-2 w-32' }, React.createElement('div', { className: 'bg-green-500 h-2 rounded-full transition-all duration-100', style: { width: holdProgress + '%' } })),
                                React.createElement('div', { key: 'progress-text', className: 'text-xs mt-1' }, 'Hold: ' + (holdProgress / 20).toFixed(1) + 's')
                            ])
                        ])
                    ]),
                    
                    React.createElement('div', { key: 'controls', className: 'flex gap-4 justify-center mb-6' }, [
                        React.createElement('button', { key: 'clear', onClick: () => setSentence(''), className: 'bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-lg font-medium transition-colors' }, 'Clear Text'),
                        React.createElement('button', { key: 'refresh', onClick: () => window.location.reload(), className: 'bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium transition-colors' }, 'Refresh')
                    ]),
                    
                    React.createElement('div', { key: 'instructions', className: 'grid md:grid-cols-2 gap-6 text-sm' }, [
                        React.createElement('div', { key: 'how-to', className: 'bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-lg' }, [
                            React.createElement('h3', { key: 'title1', className: 'text-blue-700 font-bold text-lg mb-3' }, '📝 How to Use:'),
                            React.createElement('ul', { key: 'list1', className: 'space-y-2 text-blue-800' }, [
                                React.createElement('li', {key: 'l1'}, '• Make ASL signs in front of your camera'),
                                React.createElement('li', {key: 'l2'}, '• Hold each sign steady for 5 seconds'),
                                React.createElement('li', {key: 'l3'}, '• Watch the green progress bar'),
                                React.createElement('li', {key: 'l4'}, '• Sign "space" for spaces, "del" to delete'),
                                React.createElement('li', {key: 'l5'}, '• Your sentence builds at the top')
                            ])
                        ]),
                        React.createElement('div', { key: 'tips', className: 'bg-green-50 border-l-4 border-green-500 p-4 rounded-r-lg' }, [
                            React.createElement('h3', { key: 'title2', className: 'text-green-700 font-bold text-lg mb-3' }, '🎯 Tips:'),
                            React.createElement('ul', { key: 'list2', className: 'space-y-2 text-green-800' }, [
                                React.createElement('li', {key: 't1'}, '• Keep your hand clearly visible'),
                                React.createElement('li', {key: 't2'}, '• Use good lighting'),
                                React.createElement('li', {key: 't3'}, '• Hold signs very steady'),
                                React.createElement('li', {key: 't4'}, '• Position hand in center of frame'),
                                React.createElement('li', {key: 't5'}, '• Allow camera permissions when prompted')
                            ])
                        ])
                    ])
                ]));
            };
            
            ReactDOM.render(React.createElement(ASLTranslator), document.getElementById('root'));
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        prediction, annotated_image = process_image(data['image'])
        return jsonify({'prediction': prediction, 'image': annotated_image})
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
