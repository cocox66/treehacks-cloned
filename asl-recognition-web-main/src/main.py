import sys
import time
import cv2
import argparse
import numpy as np
import mediapipe as mp
import sklearn

from autocorrect import Speller
from utils import load_model, save_gif, save_video
from utils import calc_landmark_list, draw_landmarks, draw_info_text

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Autocorrect Word
spell = Speller(lang="en")

# Colors RGB Format
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_HANDS = 1
min_detection_confidence = 0.6
min_tracking_confidence = 0.5

MODEL_PATH = "./classifier"
model_letter_path = f"{MODEL_PATH}/classify_letter_model.p"

print(f"scikit-learn version: {sklearn.__version__}")

# Customize your input
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--source", type=str, default=None, help="Video Path/0 for Webcam"
    )
    parser.add_argument(
        "-a", "--autocorrect", action="store_true", help="Autocorrect Misspelled Word"
    )
    parser.add_argument("-g", "--gif", action="store_true", help="Save GIF Result")
    parser.add_argument("-v", "--video", action="store_true", help="Save Video Result")
    parser.add_argument("-t", "--timing", type=int, default=8, help="Timing Threshold")
    parser.add_argument("-wi", "--width", type=int, default=800, help="Webcam Width")
    parser.add_argument("-he", "--height", type=int, default=600, help="Webcam Height")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Webcam FPS")
    opt = parser.parse_args()
    return opt


def get_output(idx, _output, output, autocorrect, TIMING):
    key = []
    for i in range(len(_output[idx])):
        character = _output[idx][i]
        counts = _output[idx].count(character)

        # Add character to key if it exceeds 'TIMING THRESHOLD'
        if (character not in key) or (character != key[-1]):
            if counts > TIMING:
                key.append(character)

    # Add key character to output text
    text = ""
    for character in key:
        if character == "?":
            continue
        text += str(character).lower()

    # Autocorrect Misspelled Word
    text = spell(text) if autocorrect else text

    # Add word to output list
    if text != "":
        _output[idx] = []
        output.append(text.title())
    return None


def recognize_gesture(
    image,
    results,
    model_letter_path,
    mp_drawing,
    current_hand,
    _output,
    output,
    autocorrect,
    TIMING,
):
    multi_hand_landmarks = results.multi_hand_landmarks
    multi_handedness = results.multi_handedness

    try:
        # Get our ASL recognition function
        recognize_asl = load_model(model_letter_path)
        _gesture = []

        # Draw landmarks and recognize gestures
        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            for idx in reversed(range(len(multi_hand_landmarks))):
                current_select_hand = multi_hand_landmarks[idx]
                handness = multi_handedness[idx].classification[0].label

                # Always draw landmarks
                mp_drawing.draw_landmarks(image, current_select_hand, mp_hands.HAND_CONNECTIONS)
                landmark_list = calc_landmark_list(image, current_select_hand)
                image = draw_landmarks(image, landmark_list)

                try:
                    # Recognize the gesture
                    gesture = recognize_asl(current_select_hand)
                    
                    # Draw bounding box and label
                    x_values = [lm.x for lm in current_select_hand.landmark]
                    y_values = [lm.y for lm in current_select_hand.landmark]
                    min_x = int(min(x_values) * w)
                    max_x = int(max(x_values) * w)
                    min_y = int(min(y_values) * h)
                    max_y = int(max(y_values) * h)

                    cv2.rectangle(
                        image, (min_x - 20, min_y - 10), (max_x + 20, max_y + 10), BLACK, 4
                    )
                    image = draw_info_text(
                        image, [min_x - 20, min_y - 10, max_x + 20, max_y + 10], gesture
                    )

                    _gesture.append(gesture)
                except Exception as e:
                    print(f"Error in gesture recognition: {e}")
                    continue

        # Handle output
        if len(_gesture) > 0:
            _output[0].append(_gesture[0])
            
        if results.multi_hand_landmarks:
            current_hand = len(multi_hand_landmarks)
        else:
            current_hand = 0

        return current_hand, image
    except Exception as e:
        print(f"Error in recognize_gesture: {e}")
        return current_hand, image


def recognize_signs(capture):
    current_hand = 0
    autocorrect = False
    TIMING = 8
    output = []
    _output = [[], []]
    
    with mp_hands.Hands(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        max_num_hands=MAX_HANDS,
    ) as hands:
        while capture.isOpened():
            success, image = capture.read()
            if not success:
                print("Failed to read frame.")
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                current_hand, image = recognize_gesture(
                    image,
                    results,
                    model_letter_path,
                    mp_drawing,
                    current_hand,
                    _output,
                    output,
                    autocorrect,
                    TIMING,
                )
            except Exception as error:
                _, _, exc_tb = sys.exc_info()
                print(f"{error}, line {exc_tb.tb_lineno}")

            # Show output in Top-Left corner
            output_text = str(output)
            output_size = cv2.getTextSize(output_text, FONT, 0.5, 2)[0]
            cv2.rectangle(
                image, (5, 0), (10 + output_size[0], 10 + output_size[1]), YELLOW, -1
            )
            cv2.putText(image, output_text, (10, 15), FONT, 0.5, BLACK, 2)

            cv2.imshow('MediaPipe Holistic with Speech and Vision', image)

            key = cv2.waitKey(5) & 0xFF

            # Exit conditions
            if key == 27:  # Press 'Esc' to quit
                break
            elif key == 8:  # Press 'Backspace' to delete last word
                if output:
                    output.pop()
            elif key == ord("c"):  # Press 'c' to clear output
                output.clear()
            elif output and output[-1] == "Z":  # symbol to break
                break

    return output


if __name__ == "__main__":
    opt = parse_opt()
    saveGIF = opt.gif
    saveVDO = opt.video
    source = opt.source

    _output = [[], []]
    output = []
    quitApp = False

    current_hand = 0

    global TIMING, autocorrect
    TIMING = opt.timing
    autocorrect = opt.autocorrect
    print(f"Timing Threshold is {TIMING} frames.")
    print(f"Using Autocorrect: {autocorrect}")

    capture = cv2.VideoCapture(0)
    
    out = recognize_signs(capture)
    print("HERE", out)

    cv2.destroyAllWindows()
    capture.release()
    sys.exit()
