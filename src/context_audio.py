import cv2
import speech_recognition as sr
import threading
import queue
from datetime import datetime, timedelta
from openai import OpenAI
import base64
import io
from PIL import Image
import time
import os
from dotenv import load_dotenv
import pygame
import tempfile
import sounddevice as sd
import numpy as np
import traceback
import logging
import mediapipe as mp
from utils import recognize_asl_gesture, draw_info_text

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Modify the import to include error handling
try:
    logger.debug("Attempting to import recognize_signs...")
    from main import recognize_signs
    logger.debug("Successfully imported recognize_signs")
except Exception as e:
    logger.error(f"Error importing recognize_signs: {e}")
    logger.error(traceback.format_exc())

from voice_processor import VoiceProcessor
import sys
import platform
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

text_queue = queue.Queue()
vision_queue = queue.Queue()
last_text = ""
last_vision_text = ""
text_timestamp = datetime.now()
vision_timestamp = datetime.now()
TEXT_DISPLAY_DURATION = timedelta(seconds=10)
voice_triggered = False
sign_text = ""
process_sign_language = False
voice_processor = VoiceProcessor()

# Modify the pygame initialization to handle errors gracefully
try:
    pygame.mixer.quit()  # First quit any existing mixer
    pygame.mixer.init(frequency=24000)  # Initialize with correct frequency for TTS
except Exception as e:
    logger.warning(f"Could not initialize pygame mixer: {e}")

def check_environment():
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Operating system: {platform.system()} {platform.release()}")
    logger.debug(f"Pygame version: {pygame.version.ver}")
    logger.debug(f"OpenAI API key present: {'OPENAI_API_KEY' in os.environ}")

# Add this line after load_dotenv()
check_environment()

def encode_image_to_base64(frame):
    # Convert CV2 frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image(frame):
    logger.debug("analyze_image")
    try:
        base64_image = encode_image_to_base64(frame)
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "As a friendly companion, describe what's immediately around us in 1-2 short sentences. Focus on the most important things: any nearby obstacles, people, or immediate safety concerns a visually impaired person should know about. Don't say more than 3 sentences."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ],
                }
            ],
            max_tokens=100  # Reduced token limit to ensure shorter responses
        )
        
        vision_queue.put(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")

# Modify the audio device listing to handle errors
def list_all_audio_devices():
    try:
        logger.debug("\nAvailable Audio Input Devices:")
        logger.debug("------------------------------")
        devices = sr.Microphone.list_microphone_names()
        for index, name in enumerate(devices):
            logger.debug(f"Index {index}: {name}")
        logger.debug("------------------------------")
    except Exception as e:
        logger.warning(f"Could not list audio devices: {e}")

# Add this line after pygame.mixer.init()
list_all_audio_devices()


def audio_processing():
    recognizer = sr.Recognizer()
    mic_index = 0 #0 for MAC MIC, 3 for MAC SPEAKER TO MIC 
    
    while True:
        try:
            with sr.Microphone(device_index=mic_index) as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                try:
                    text = recognizer.recognize_google(audio)
                    logger.debug("received text")
                    # Check for trigger phrases
                    trigger_phrases = ["describe surround", "describe my surroundings", "describe surrounding", "describe surroundings", "what's around", "whats around", "describe the room"]

                    if "sign language" in text.lower():
                        logger.debug("recognized sign language phrase")
                        text_queue.put("Recognizing sign language...")
                        global sign_text, process_sign_language
                        process_sign_language = True  # New flag to trigger sign language processing
                    elif any(phrase in text.lower() for phrase in trigger_phrases):
                        text_queue.put("Analyzing surroundings...")
                        global voice_triggered
                        voice_triggered = True
                    else:
                        text_queue.put(f"Speech: {text}")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    text_queue.put("Speech recognition service unavailable")
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")

# Start audio processing thread
audio_thread = threading.Thread(target=audio_processing, daemon=True)
audio_thread.start()

# Initialize video capture
cap = cv2.VideoCapture(1)  #0 for MAC CAMERA, 1 for OBS CAMERA

if not cap.isOpened():
    logger.warning("Failed to open camera 2, trying camera 1...")
    cap = cv2.VideoCapture(1)  # Try index 1 as fallback

if not cap.isOpened():
    logger.warning("Failed to open OBS camera, falling back to default camera...")
    cap = cv2.VideoCapture(0)  # Fallback to default camera

logger.debug(f"Successfully opened camera with index: {cap.get(cv2.CAP_PROP_POS_FRAMES)}")

# Modify the text_to_speech function to be more robust
def text_to_speech(text):
    try:
        # Use sounddevice directly without pygame
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
            response_format="pcm",
            speed=1.0
        )
        
        audio_data = np.frombuffer(response.content, dtype=np.int16)
        
        # Add error checking for audio playback
        try:
            sd.stop()  # Stop any currently playing audio
            sd.play(audio_data, samplerate=24000)
            sd.wait()
        except sd.PortAudioError as e:
            logger.error(f"Audio playback error: {e}")
            
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {e}")

def process_frame_with_audio(frame, hands, hand_connections, process_sign_language, voice_triggered):
    """Process a single frame with ASL recognition and audio features"""
    try:
        # Convert frame color for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        text_output = ""
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, hand_connections)
                
                if process_sign_language:
                    # Get ASL gesture
                    gesture = recognize_asl_gesture(hand_landmarks)
                    
                    # Draw bounding box and label
                    h, w, _ = image.shape
                    x_values = [lm.x for lm in hand_landmarks.landmark]
                    y_values = [lm.y for lm in hand_landmarks.landmark]
                    min_x = int(min(x_values) * w)
                    max_x = int(max(x_values) * w)
                    min_y = int(min(y_values) * h)
                    max_y = int(max(y_values) * h)
                    
                    cv2.rectangle(
                        image, 
                        (min_x - 20, min_y - 10), 
                        (max_x + 20, max_y + 10), 
                        (0, 0, 0), 
                        4
                    )
                    
                    image = draw_info_text(
                        image,
                        [min_x - 20, min_y - 10, max_x + 20, max_y + 10],
                        gesture
                    )
                    
                    text_output = gesture
        
        # Handle voice trigger if needed
        if voice_triggered:
            # Add any voice processing logic here
            pass
            
        return image, text_output
        
    except Exception as e:
        print(f"Error in process_frame_with_audio: {e}")
        return frame, ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update transcribed text if available
    try:
        while not text_queue.empty():
            last_text = text_queue.get_nowait()
            text_timestamp = datetime.now()
    except queue.Empty:
        pass
    if process_sign_language:
        try:
            logger.debug("Starting sign language recognition...")
            sign_text = recognize_signs(cap)  # Pass the existing camera capture object
            logger.debug(f"Sign recognition result: {sign_text}")
            if sign_text:  # Only process if we got a result
                threading.Thread(
                    target=lambda: voice_processor.text_to_speech(
                        voice_processor.correct_sign_output(" ".join(sign_text))
                    ),
                    daemon=True
                ).start()
        except Exception as e:
            logger.error(f"Error in sign language recognition: {e}")
            logger.error(traceback.format_exc())
        finally:
            process_sign_language = False  # Reset the flag

    # Only analyze image when voice triggered
    if voice_triggered:
        threading.Thread(target=analyze_image, args=(frame.copy(),), daemon=True).start()
        voice_triggered = False  # Reset the trigger

    # Update vision analysis text if available
    try:
        while not vision_queue.empty():
            last_vision_text = vision_queue.get_nowait()
            vision_timestamp = datetime.now()
            threading.Thread(target=text_to_speech, args=(last_vision_text,), daemon=True).start()
    except queue.Empty:
        pass

    # Draw existing landmarks
    # ... existing landmark drawing code ...

    # Add text overlays
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create overlay for better text visibility
    overlay = frame.copy()
    
    # Speech transcription overlay (bottom)
    if datetime.now() - text_timestamp < TEXT_DISPLAY_DURATION:
        cv2.rectangle(overlay, (10, frame_height-60), 
                     (frame_width-10, frame_height-10), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, last_text, 
                    (20, frame_height-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (255, 255, 255), 2)

    # Vision analysis overlay (top)
    if datetime.now() - vision_timestamp < TEXT_DISPLAY_DURATION:
        # Split text into multiple lines if too long
        words = last_vision_text.split()
        lines = []
        current_line = "Vision: "
        for word in words:
            if len(current_line + word) < 50:  # Adjust number based on your needs
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)

        # Draw background for vision text
        cv2.rectangle(overlay, (10, 10), 
                     (frame_width-10, 20 + 30*len(lines)), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Draw each line of text
        for i, line in enumerate(lines):
            cv2.putText(frame, line, 
                        (20, 40 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (255, 255, 255), 2)

    cv2.imshow('MediaPipe Holistic with Speech and Vision', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()