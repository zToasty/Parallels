import cv2
import requests
import time
import numpy as np

SERVER_URL = "http://<Server-IP>:44443/upload"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру.")
    exit()

frame_count = 0
start_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр.")
        break

    # Кодируем кадр в jpg
    _, img_encoded = cv2.imencode('.jpg', frame)
    try:
        response = requests.post(SERVER_URL, data=img_encoded.tobytes(), timeout=2)
        if response.status_code == 200:

            nparr = np.frombuffer(response.content, np.uint8)
            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if processed_frame is not None:
                
                cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Processed Frame', processed_frame)
        else:
            print(f"Ошибка: {response.status_code} — {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[Request Error] {e}")

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
