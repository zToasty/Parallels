from flask import Flask, request, Response
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# --- Ring Buffer ---
class RingBuffer:
    def __init__(self, size):
        self.buffer = [None] * size
        self.size = size
        self.read_idx = 0
        self.write_idx = 0
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            self.buffer[self.write_idx] = item
            self.write_idx = (self.write_idx + 1) % self.size
            if self.write_idx == self.read_idx:
                self.read_idx = (self.read_idx + 1) % self.size  # overwrite

    def get(self):
        with self.lock:
            if self.buffer[self.read_idx] is None:
                return None
            item = self.buffer[self.read_idx]
            self.buffer[self.read_idx] = None
            self.read_idx = (self.read_idx + 1) % self.size
            return item

# --- Video Processor ---
class VideoProcessor:
    def __init__(self, buffer_size=60, num_workers=4):
        self.capture_buffer = RingBuffer(buffer_size)
        self.processed_frame = None
        self.processed_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        for _ in range(num_workers):
            self.executor.submit(self.processing_worker)

    def processing_worker(self):
        model = YOLO("yolov8s-pose.pt")
        while not self.stop_event.is_set():
            frame = self.capture_buffer.get()
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                result = model.predict(frame, device='cpu')[0]
                processed = result.plot()
                with self.processed_lock:
                    self.processed_frame = processed
            except Exception as e:
                print(f"[ERROR] Processing: {e}")

    def add_frame(self, frame):
        self.capture_buffer.put(frame)

    def get_latest_frame(self):
        with self.processed_lock:
            return self.processed_frame.copy() if self.processed_frame is not None else None

    def shutdown(self):
        self.stop_event.set()
        self.executor.shutdown(wait=True)

# --- Flask App ---
app = Flask(__name__)
video_processor = VideoProcessor(buffer_size=60, num_workers=4)

FRAME_INTERVAL = 1 / 60
last_upload_time = 0
upload_lock = threading.Lock()

@app.route('/upload', methods=['POST'])
def upload_frame():
    global last_upload_time
    with upload_lock:
        now = time.time()
        if now - last_upload_time < FRAME_INTERVAL:
            return Response("Too fast", status=429)
        last_upload_time = now

    try:
        nparr = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return Response("Invalid image", status=400)

        video_processor.add_frame(frame)

        timeout = time.time() + 1.0
        while time.time() < timeout:
            processed = video_processor.get_latest_frame()
            if processed is not None:
                _, img_encoded = cv2.imencode('.jpg', processed)
                return Response(img_encoded.tobytes(), mimetype='image/jpeg')
            time.sleep(0.01)

        return Response("No processed frame", status=504)

    except Exception as e:
        print(f"Upload error: {e}")
        return Response("Server error", status=500)

# --- Shutdown Hook ---
import atexit
@atexit.register
def shutdown():
    video_processor.shutdown()

# --- Run Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=44443)