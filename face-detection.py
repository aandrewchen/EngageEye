import cv2
import mediapipe as mp
import time
from datetime import datetime, timedelta

cap = cv2.VideoCapture(1)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mpFaceDetection.FaceDetection(0.75)

last_eyes_seen = datetime.now()

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mpFaceDetection.process(imgRGB)

    eyes_detected = False

    if results.detections:
        for detection in results.detections:
            keypointsC = detection.location_data.relative_keypoints
            h, w, c = img.shape
            for i in range(2):
                eyes = int(keypointsC[i].x * w), int(keypointsC[i].y * h)
                cv2.circle(img, eyes, 10, (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (eyes[0], eyes[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                if detection.score[0] > 0.8:
                    eyes_detected = True

    if eyes_detected:
        last_eyes_seen = datetime.now()
    elif datetime.now() - last_eyes_seen > timedelta(seconds=10):
        text = "Eyes not detected!"
        font_scale = 5
        thickness = 5
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), thickness)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()