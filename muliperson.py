import cv2
import mediapipe as mp

# Mediapipeの設定
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Webカメラの設定
cap = cv2.VideoCapture(0)

# Handsソリューションの初期化（max_num_handsを必要に応じて設定）
with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=4,   # 必要な手の数に応じて設定
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラからフレームを取得できませんでした")
            continue

        # 映像を左右反転
        image = cv2.flip(image, 1)

        # BGRからRGBに変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 手検出の処理
        results = hands.process(image)

        # 描画用にRGBからBGRに変換
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 検出された手のランドマークを描画
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # cv2.circle(frame, (int(index_finger_tip.x * w), int(index_finger_tip.y * h)), 5, (0, 0, 255), cv2.FILLED)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 画面に表示
        cv2.imshow('Hand Detection (Mirrored)', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
