import cv2
import pickle
import string
import imageio
import numpy as np
import mediapipe as mp

# 26 Labels and Unknown Gesture
ascii_string = string.ascii_lowercase.upper() + "?"
labels_dict = {idx: value for idx, value in enumerate(ascii_string)}

# Colors RGB Format
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


def draw_info_text(image, pos, hand_sign_text):
    """Draw the recognized gesture text on the image"""
    # Draw background rectangle
    cv2.rectangle(image, 
                 (pos[0] - 2, pos[1] - 30), 
                 (pos[2] + 2, pos[1] - 2), 
                 (0, 0, 0), 
                 -1)
    
    # Draw text
    cv2.putText(
        image,
        f"Sign: {hand_sign_text}",
        (pos[0] + 5, pos[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    return image


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), WHITE, 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), WHITE, 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), WHITE, 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), WHITE, 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), WHITE, 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), WHITE, 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), WHITE, 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 1:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 2:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 3:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 4:
            cv2.circle(image, (landmark[0], landmark[1]), 8, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, BLACK, 1)
        if index == 5:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 6:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 7:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 8:
            cv2.circle(image, (landmark[0], landmark[1]), 8, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, BLACK, 1)
        if index == 9:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 10:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 11:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 12:
            cv2.circle(image, (landmark[0], landmark[1]), 8, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, BLACK, 1)
        if index == 13:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 14:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 15:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 16:
            cv2.circle(image, (landmark[0], landmark[1]), 8, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, BLACK, 1)
        if index == 17:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 18:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 19:
            cv2.circle(image, (landmark[0], landmark[1]), 5, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, BLACK, 1)
        if index == 20:
            cv2.circle(image, (landmark[0], landmark[1]), 8, WHITE, -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, BLACK, 1)

    return image


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    # Keypoint
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def classify_landmark(landmark):
    wrist = landmark[0]
    thump = landmark[1:5]
    index_finger = landmark[5:9]
    middle_finger = landmark[9:13]
    ring_finger = landmark[13:17]
    pinky = landmark[17:21]
    return [[wrist], thump, index_finger, middle_finger, ring_finger, pinky]


def is_finger_on(idx, finger, landmark_label):
    if idx == 0:
        if landmark_label == "Right":
            return finger[-1].x < finger[-2].x
        else:
            return finger[-1].x > finger[-2].x
    return finger[-1].y < finger[0].y


def save_gif(gif_array, fps=30, output_dir="./assets/result.gif"):
    duration = 1000 / fps

    # Convert to gif using the imageio.mimsave method
    imageio.mimwrite(
        f"{output_dir}", gif_array, duration=duration, format="GIF", loop=0
    )
    print(f"Save to {output_dir}!")


def save_video(frame_array, width, height, fps=30, output_dir="./assets/result.mp4"):
    output_writer = cv2.VideoWriter(
        f"{output_dir}", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for frame in frame_array:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_writer.write(frame)

    output_writer.release()
    print(f"Save to {output_dir}!")


def recognize_asl_gesture(landmarks):
    """
    Recognize ASL gestures based on finger positions
    """
    # Convert landmarks to relative positions
    points = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
    
    # Define finger states
    thumb_tip = points[4]
    thumb_ip = points[3]
    index_tip = points[8]
    index_pip = points[6]
    middle_tip = points[12]
    middle_pip = points[10]
    ring_tip = points[16]
    ring_pip = points[14]
    pinky_tip = points[20]
    pinky_pip = points[18]

    # Check if fingers are extended (comparing y coordinates)
    thumb_extended = thumb_tip[0] < thumb_ip[0]  # For right hand
    index_extended = index_tip[1] < index_pip[1]
    middle_extended = middle_tip[1] < middle_pip[1]
    ring_extended = ring_tip[1] < ring_pip[1]
    pinky_extended = pinky_tip[1] < pinky_pip[1]

    # Basic ASL letter recognition
    if not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return 'A'
    elif all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
        return 'B'
    elif thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return 'L'
    elif thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        return 'Y'
    elif index_extended and not any([thumb_extended, middle_extended, ring_extended, pinky_extended]):
        return 'D'
    elif all([index_extended, middle_extended]) and not any([thumb_extended, ring_extended, pinky_extended]):
        return 'U'
    elif all([thumb_extended, index_extended, pinky_extended]) and not any([middle_extended, ring_extended]):
        return 'I LOVE YOU'
    
    return '?'


def load_model(model_path):
    """
    Instead of loading a pickle model, we'll use MediaPipe's built-in hand detection
    and our simplified ASL recognition
    """
    # print("Using MediaPipe hand detection with basic ASL recognition")
    return recognize_asl_gesture  # Return the function itself

