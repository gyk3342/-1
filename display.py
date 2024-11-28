import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
import datetime

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=5, min_detection_confidence=0.7, min_tracking_confidence=0.7)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
left_hand_ids = {}
right_hand_ids = {}

canvas = None
canvas_size_multiplier = 3  
smoothing_queue = {}  
smoothing_length = 5  

drawing_mode = 'pen'  
line_thickness = 5 
color_index = 0 
save_directory = "./saved_drawings" 
drawing_enabled = True 

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

def toggle_drawing_mode():
    global drawing_mode
    if drawing_mode == 'pen':
        drawing_mode = 'eraser'
    else:
        drawing_mode = 'pen'

cap = cv2.VideoCapture(0)
tips_previous_positions = {}  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        h, w, _ = frame.shape
        canvas = np.ones((h * canvas_size_multiplier, w * canvas_size_multiplier, 3), dtype=np.uint8) * 255

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)  

    if results.multi_hand_landmarks:
        detected_hands = []
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_idx].classification[0].label
            hand_label = f'{handedness}_{hand_idx}'
            if handedness == 'Left':
                if hand_label not in left_hand_ids:
                    left_hand_ids[hand_label] = colors[len(left_hand_ids) % len(colors)]  
                color = left_hand_ids[hand_label]
            else:
                if hand_label not in right_hand_ids:
                    right_hand_ids[hand_label] = colors[len(right_hand_ids) % len(colors)] 
                color = right_hand_ids[hand_label]

            palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            if palm_base.y > 0.98:  
                continue

            detected_hands.append(hand_label)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w * canvas_size_multiplier), int(index_finger_tip.y * h * canvas_size_multiplier)

            if hand_label not in smoothing_queue:
                smoothing_queue[hand_label] = deque(maxlen=smoothing_length)
            smoothing_queue[hand_label].append((cx, cy))

            smoothed_cx = int(np.mean([p[0] for p in smoothing_queue[hand_label]]))
            smoothed_cy = int(np.mean([p[1] for p in smoothing_queue[hand_label]]))

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
            distance_thumb_index = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
            distance_index_middle = np.sqrt((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2)

            if distance_thumb_index < 30: 
                drawing_enabled = False
            else: 
                drawing_enabled = True

            if distance_index_middle < 30:
                drawing_mode = 'eraser'
            else:
                drawing_mode = 'pen'

            if drawing_enabled and hand_label in tips_previous_positions:
                px, py = tips_previous_positions[hand_label]
                if drawing_mode == 'pen':
                    cv2.line(canvas, (px, py), (smoothed_cx, smoothed_cy), color, line_thickness)
                elif drawing_mode == 'eraser':
                    cv2.line(canvas, (px, py), (smoothed_cx, smoothed_cy), (255, 255, 255), line_thickness * 20)

            tips_previous_positions[hand_label] = (smoothed_cx, smoothed_cy)

            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            distance_text = f"{hand_label} distance: {int(distance_thumb_index)}"
            text_y_position = 50 + hand_idx * 30  
            cv2.putText(frame, distance_text, (10, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        tips_previous_positions = {hand: pos for hand, pos in tips_previous_positions.items() if hand in detected_hands}
    else:
        tips_previous_positions.clear()

    resized_canvas = cv2.resize(canvas, (w, h))
    combined_frame = cv2.addWeighted(frame, 0.5, resized_canvas, 0.5, 0)  

    cv2.imshow('multi_painting', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_directory, f"drawing_{timestamp}.png")
        cv2.imwrite(save_path, canvas)
        print(f"save: {save_path}")
    elif key == ord('c'):
        canvas = np.ones((h * canvas_size_multiplier, w * canvas_size_multiplier, 3), dtype=np.uint8) * 255
        print("clear")

cap.release()
cv2.destroyAllWindows()
