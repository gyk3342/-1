import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=5)
mp_drawing = mp.solutions.drawing_utils

canvas = None
canvas_size_multiplier = 2

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
hand_ids = {} 

cap = cv2.VideoCapture(0)
tips_previous_positions = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    if canvas is None:
        h, w, _ = frame.shape
        canvas = np.zeros((h * canvas_size_multiplier, w * canvas_size_multiplier, 3), dtype=np.uint8)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[hand_idx].classification[0].label
            if hand_label not in hand_ids:
                hand_ids[hand_label] = colors[len(hand_ids) % len(colors)]
                
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w * canvas_size_multiplier), int(index_finger_tip.y * h * canvas_size_multiplier)

            color = hand_ids[hand_label]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            line_thickness = int(1000 / (distance + 1))
            line_thickness = np.clip(line_thickness, 1, 20)

            if hand_label in tips_previous_positions:
                px, py = tips_previous_positions[hand_label]
                cv2.line(canvas, (px, py), (cx, cy), color, line_thickness)

            tips_previous_positions[hand_label] = (cx, cy)
            cv2.circle(frame, (int(index_finger_tip.x * w), int(index_finger_tip.y * h)), 5, (0, 0, 255), cv2.FILLED)
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            distance_text = f"{hand_label} Distance: {int(distance)}"
            text_y_position = 50 + hand_idx * 30 
            cv2.putText(frame, distance_text, (10, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    else:
        tips_previous_positions.clear()

    resized_canvas = cv2.resize(canvas, (w, h))
    combined_frame = cv2.addWeighted(frame, 0.5, resized_canvas, 0.5, 0)

    cv2.imshow('Multi-User Gesture Drawing', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

