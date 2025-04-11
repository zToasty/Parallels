import cv2 as cv
import time
import threading
import queue
import argparse
import logging
import os
import sys

log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, "sensor_log.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    """Simulated sensor with delay"""
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    """Camera sensor"""
    def __init__(self, camera_name, resolution):
        self.camera_name = camera_name
        self.width, self.height = map(int, resolution.split('x'))
        try:
            self.cap = cv.VideoCapture(int(camera_name) if camera_name.isdigit() else camera_name, cv.CAP_DSHOW)
            if not self.cap.isOpened():
                raise ValueError("Camera not found")
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception as e:
            logging.error(f"Fatal error: Camera initialization failed - {e}")
            sys.exit(1)

    def get(self):
        """Retrieve a frame from the camera"""
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Camera disconnected or failed to read frame")
            return None
        return frame

    def release(self):
        if hasattr(self, "cap"):
            self.cap.release()

    def __del__(self):
        self.release()


class WindowImage:
    """Window display handler"""
    def __init__(self, display_rate):
        self.display_rate = display_rate

    def show(self, img):
        """Display the frame"""
        cv.imshow("Sensor Data", img)
        if cv.waitKey(int(1000 / self.display_rate)) & 0xFF == ord('q'):
            return False
        return True

    def release(self):
        cv.destroyAllWindows()

    def __del__(self):
        self.release()


def sensor_worker(sensor, data_queue):
    """Thread for reading sensor data"""
    while True:
        try:
            value = sensor.get()
            if data_queue.full():
                data_queue.get()
            data_queue.put(value)
        except Exception as e:
            logging.error(f"Sensor error: {e}")
            time.sleep(1)


def cleanup(camera, sensors, window):
    """Release all resources"""
    logging.info("Cleaning up resources...")
    if camera:
        camera.release()
    for sensor in sensors:
        del sensor
    if window:
        window.release()

def camera_worker(camera, frame_queue, stop_event):
    """Thread for capturing camera frames"""
    while not stop_event.is_set():
        try:
            frame = camera.get()
            if frame is None:
                logging.error("Camera frame is None, possible disconnection")
                stop_event.set()
                return
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(frame)
        except Exception as e:
            logging.error(f"Fatal camera worker error: {e}")
            stop_event.set()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="0", help="Camera name or ID")
    parser.add_argument("--resolution", type=str, default="640x480", help="Camera resolution")
    parser.add_argument("--fps", type=int, default=30, help="Display refresh rate (FPS)")
    args = parser.parse_args()

    camera = None
    sensors = []
    window = None
    stop_event = threading.Event()

    try:
        camera = SensorCam(args.camera, args.resolution)
        sensors = [SensorX(0.01), SensorX(0.1), SensorX(1)]
        frame_queue = queue.Queue(maxsize=2)
        sensor_queues = [queue.Queue(maxsize=2) for _ in sensors]

        # Sensor threads
        for sensor, sensor_queue in zip(sensors, sensor_queues):
            threading.Thread(target=sensor_worker, args=(sensor, sensor_queue), daemon=True).start()

        # Camera thread
        cam_thread = threading.Thread(target=camera_worker, args=(camera, frame_queue, stop_event), daemon=True)
        cam_thread.start()

        window = WindowImage(args.fps)

        # FPS sync
        prev_time = time.time()

        while not stop_event.is_set():
            current_time = time.time()
            elapsed_time = current_time - prev_time

            
            if elapsed_time >= (1 / args.fps):
                prev_time = current_time

                try:
                    frame = frame_queue.get_nowait()
                except queue.Empty:
                    frame = None

                if frame is not None:
                    for i, sensor_queue in enumerate(sensor_queues):
                        try:
                            sensor_value = sensor_queue.get_nowait()
                        except queue.Empty:
                            sensor_value = "N/A"
                        cv.putText(frame, f"Sensor {i}: {sensor_value}", (10, 30 + i * 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if not window.show(frame):
                        logging.info("KeyboardInterrupt: Exiting program")
                        stop_event.set()
                        break
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: Exiting program")
    except SystemExit:
        logging.error("System exit triggered due to fatal error")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        cleanup(camera, sensors, window)
        logging.info("Program exited successfully.")
        sys.exit(0)