import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
import datetime

# Mediapipe Hands设置，用于手部检测和关键点识别
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 初始化Mediapipe Hands，支持多个用户（最多5只手），提高检测和跟踪的置信度
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=5, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 用于区分不同手的颜色，每只手分配不同的颜色
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
left_hand_ids = {}
right_hand_ids = {}

# 绘画用的画布设置，初始为空
canvas = None
canvas_size_multiplier = 3  # 画布大小是视频帧的三倍，以便绘制更广泛的内容
smoothing_queue = {}  # 用于存储每只手的轨迹，进行平滑处理
smoothing_length = 5  # 平滑队列的长度

drawing_mode = 'pen'  # 默认为画笔模式，其他选项可以是橡皮擦、矩形等
line_thickness = 5  # 默认线条粗细
color_index = 0  # 用于选择当前颜色
save_directory = "./saved_drawings"  # 保存绘画的文件夹路径
drawing_enabled = True  # 控制是否绘画的标志

# 如果保存绘画的文件夹不存在，则创建该文件夹
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

def toggle_drawing_mode():
    global drawing_mode
    if drawing_mode == 'pen':
        drawing_mode = 'eraser'
    else:
        drawing_mode = 'pen'

# 初始化视频捕捉，从默认摄像头捕捉视频流
cap = cv2.VideoCapture(0)
tips_previous_positions = {}  # 用于存储每只手的食指指尖的上一个位置

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 水平翻转帧以获得镜像视图，使视频看起来更自然
    frame = cv2.flip(frame, 1)

    # 如果画布尚未初始化，则根据当前帧的尺寸创建画布
    if canvas is None:
        h, w, _ = frame.shape
        canvas = np.ones((h * canvas_size_multiplier, w * canvas_size_multiplier, 3), dtype=np.uint8) * 255

    # 将帧的颜色空间从BGR转换为RGB格式，以供Mediapipe使用
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)  # 使用Mediapipe处理当前帧，检测手部并获取关键点

    # 处理检测到的手部标志点
    if results.multi_hand_landmarks:
        detected_hands = []
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 获取当前手的唯一ID，并根据左右手进行区分
            handedness = results.multi_handedness[hand_idx].classification[0].label
            hand_label = f'{handedness}_{hand_idx}'
            if handedness == 'Left':
                if hand_label not in left_hand_ids:
                    left_hand_ids[hand_label] = colors[len(left_hand_ids) % len(colors)]  # 给每只左手分配一个颜色
                color = left_hand_ids[hand_label]
            else:
                if hand_label not in right_hand_ids:
                    right_hand_ids[hand_label] = colors[len(right_hand_ids) % len(colors)]  # 给每只右手分配一个颜色
                color = right_hand_ids[hand_label]

            # 只处理在屏幕合理区域内的手，防止误识别边缘的手
            palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            if palm_base.y > 0.98:  # 如果手腕位于帧的最底部区域，忽略该手
                continue

            detected_hands.append(hand_label)

            # 获取拇指指尖、食指指尖和中指指尖的关键点坐标
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # 计算画布上的坐标，以便绘制手势轨迹
            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w * canvas_size_multiplier), int(index_finger_tip.y * h * canvas_size_multiplier)

            # 平滑轨迹，使用队列存储最近的点
            if hand_label not in smoothing_queue:
                smoothing_queue[hand_label] = deque(maxlen=smoothing_length)
            smoothing_queue[hand_label].append((cx, cy))

            # 计算平滑后的坐标
            smoothed_cx = int(np.mean([p[0] for p in smoothing_queue[hand_label]]))
            smoothed_cy = int(np.mean([p[1] for p in smoothing_queue[hand_label]]))

            # 判断是否握拳来停止绘画，或者伸出手指来开始绘画
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
            distance_thumb_index = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
            distance_index_middle = np.sqrt((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2)

            if distance_thumb_index < 30:  # 握拳时停止绘画
                drawing_enabled = False
            else:  # 伸出手指时开始绘画
                drawing_enabled = True

            # 双指并拢时切换到橡皮擦模式
            if distance_index_middle < 30:
                drawing_mode = 'eraser'
            else:
                drawing_mode = 'pen'

            # 如果有之前记录的位置，并且绘画功能启用，则绘制线条，形成绘画效果
            if drawing_enabled and hand_label in tips_previous_positions:
                px, py = tips_previous_positions[hand_label]
                if drawing_mode == 'pen':
                    cv2.line(canvas, (px, py), (smoothed_cx, smoothed_cy), color, line_thickness)
                elif drawing_mode == 'eraser':
                    cv2.line(canvas, (px, py), (smoothed_cx, smoothed_cy), (255, 255, 255), line_thickness * 20)

            # 更新当前食指指尖的位置，供下一帧使用
            tips_previous_positions[hand_label] = (smoothed_cx, smoothed_cy)

            # 在帧上绘制手部的关键点和连接线，方便用户查看手部的识别情况
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 在帧上显示拇指和食指之间的距离信息，便于用户理解当前的手势状态
            distance_text = f"{hand_label} distance: {int(distance_thumb_index)}"
            text_y_position = 50 + hand_idx * 30  # 每只手的信息显示在不同的位置
            cv2.putText(frame, distance_text, (10, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 清除未检测到的手的历史位置，防止手的误识别和轨迹连线问题
        tips_previous_positions = {hand: pos for hand, pos in tips_previous_positions.items() if hand in detected_hands}
    else:
        # 如果未检测到手，则清除之前的位置，以免误绘
        tips_previous_positions.clear()

    # 调整画布大小，使其与视频帧大小匹配，然后与视频帧进行合并
    resized_canvas = cv2.resize(canvas, (w, h))
    combined_frame = cv2.addWeighted(frame, 0.5, resized_canvas, 0.5, 0)  # 将帧和画布以一定的权重进行混合，形成最终输出

    # 显示合并后的帧，包含原始视频和绘画内容
    cv2.imshow('multi_painting', combined_frame)

    # 检测按键事件
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # 保存当前绘画内容
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_directory, f"drawing_{timestamp}.png")
        cv2.imwrite(save_path, canvas)
        print(f"save: {save_path}")
    elif key == ord('c'):
        # 清空画布
        canvas = np.ones((h * canvas_size_multiplier, w * canvas_size_multiplier, 3), dtype=np.uint8) * 255
        print("clear")

# 释放视频捕捉并销毁所有窗口，清理资源
cap.release()
cv2.destroyAllWindows()
