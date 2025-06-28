from flask import Flask, Response, render_template
from flask_cors import CORS
import cv2
import mediapipe as mp
import queue
import logging
from dotenv import load_dotenv
from utils import recognize_asl_gesture, draw_info_text, draw_landmarks, calc_landmark_list
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
stop_threads = False
process_sign_language = True  # Set to True by default
voice_triggered = False
current_gesture = ""

def initialize_camera():
    """Initialize camera with fallback options"""
    # Try OBS virtual camera first
    capture = cv2.VideoCapture(1)
    if not capture.isOpened():
        logger.warning("Failed to open OBS camera (index 1), trying default camera...")
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            logger.error("Failed to open any camera")
            return None
    logger.info(f"Successfully opened camera")
    return capture

def process_hand_landmarks(image, hand_landmarks, process_sign_language):
    """Process hand landmarks and return gesture if enabled"""
    global current_gesture
    
    h, w, _ = image.shape
    
    # Calculate landmark points for drawing
    landmark_list = calc_landmark_list(image, hand_landmarks)
    
    # Draw the landmarks with our custom drawing function
    image = draw_landmarks(image, landmark_list)
    
    if process_sign_language:
        try:
            # Get ASL gesture using our recognition function
            gesture = recognize_asl_gesture(hand_landmarks)
            current_gesture = gesture
            
            # Draw bounding box and label
            x_values = [lm.x for lm in hand_landmarks.landmark]
            y_values = [lm.y for lm in hand_landmarks.landmark]
            min_x = int(min(x_values) * w)
            max_x = int(max(x_values) * w)
            min_y = int(min(y_values) * h)
            max_y = int(max(y_values) * h)
            
            # Draw rectangle
            cv2.rectangle(
                image, 
                (min_x - 20, min_y - 10), 
                (max_x + 20, max_y + 10), 
                (0, 255, 0),  # Green color
                2
            )
            
            # Draw gesture label
            image = draw_info_text(
                image,
                [min_x - 20, min_y - 10, max_x + 20, max_y + 10],
                gesture
            )
            
            # Draw current gesture in top-left corner
            cv2.putText(
                image,
                f"Current Sign: {gesture}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),  # Green color
                2,
                cv2.LINE_AA
            )
            
            logger.info(f"Recognized gesture: {gesture}")
                
        except Exception as e:
            logger.error(f"Error in gesture recognition: {e}")
            logger.error(traceback.format_exc())
    
    return image

def generate_frames():
    """Generate processed frames for video streaming"""
    global process_sign_language
    
    capture = initialize_camera()
    if capture is None:
        return
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        max_num_hands=1
    )
    
    while not stop_threads:
        success, frame = capture.read()
        if not success:
            logger.warning("Failed to read frame")
            continue
            
        try:
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw hand landmarks and process gestures if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    image = process_hand_landmarks(
                        image, 
                        hand_landmarks, 
                        process_sign_language
                    )
            
            # Add status indicator
            status_text = "RECOGNITION: ON" if process_sign_language else "RECOGNITION: OFF"
            status_color = (0, 255, 0) if process_sign_language else (0, 0, 255)  # Green for ON, Red for OFF
            
            # Draw status with background
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                image,
                (10, image.shape[0] - 40),
                (10 + text_size[0] + 20, image.shape[0] - 10),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                image,
                status_text,
                (20, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA
            )
            
            # Convert to jpg for streaming
            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            continue
    
    capture.release()

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/toggle_sign_language', methods=['POST'])
def toggle_sign_language():
    global process_sign_language
    process_sign_language = not process_sign_language
    return {'status': 'success', 'processing': process_sign_language}

@app.route('/get_current_gesture', methods=['GET'])
def get_current_gesture():
    return {'gesture': current_gesture}

@app.route('/trigger_voice', methods=['POST'])
def trigger_voice():
    global voice_triggered
    voice_triggered = True
    return {'status': 'success'}

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        app.run(debug=True, port=5000)
    except Exception as e:
        logger.error(f"Server error: {e}") 