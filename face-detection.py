import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mpFaceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            keypointsC = detection.location_data.relative_keypoints
            h, w, c = img.shape
            for i in range(2):
                keypoints = int(keypointsC[i].x * w), int(keypointsC[i].y * h)
                cv2.circle(img, keypoints, 10, (255, 0, 255), 2)
            print(id, detection)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()