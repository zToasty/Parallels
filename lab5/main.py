from ultralytics import YOLO
import threading
import queue
from queue import Queue
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
import time
import argparse

# разделение видео на кадры и закидывание их в очередь для последующей обработки нейросеткой
def video_prepare(video_queue:Queue,video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened:
        print("не открылось")
        cap.release()
    c  = 0
    while True:

        ret,frame = cap.read()

        if not ret:break
        video_queue.put((frame,c))
        c+=1 

    cap.release()

    return c , fps

class VideoPredictor:
    def __init__(self,video_path,video_queue:Queue,num_threads=1):
        self.video_path = video_path
        self.video_queue = video_queue
        self.num_threads = num_threads
        self.queue_lock = threading.Lock()
        self.map_lock = threading.Lock()
        self.frames = {}
        self.__stop_event = threading.Event()
        self.workers = []

 
    def thread_safe_predict(self):
        local_model = YOLO('yolov8s-pose.pt')
        while True:
            try:
                with self.queue_lock:
                    frame,idx = self.video_queue.get(timeout=1)
                res = local_model.predict(frame,device='cpu')
                with self.map_lock:
                    self.frames[idx] = res[0].plot()
            except queue.Empty:
                    if self.__stop_event.is_set():
                        print(f'работа завершена by thread {threading.get_ident()}')
                        break

    def video_decode(self,fps,c,name):
        height,width,_ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{name}.mp4",fourcc,fps,(width,height))
        for idx in range(c):
            out.write(self.frames[idx])
        out.release()
    
    
    def start(self):
        for _ in range(self.num_threads):
            self.workers.append(threading.Thread(target=self.thread_safe_predict))
        for thr in self.workers:
            thr.start()

    def stop(self):
        for thr in self.workers:
            self.__stop_event.set()
            thr.join()


    def __del__(self):
        self.stop()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="путь к видео,количество потоков,название итогового видео")
    parser.add_argument("--videopath", type=str, default='input.avi')
    parser.add_argument("--numthreads", type=int, default=1)
    parser.add_argument("--name", type=str, default='output')
    args = parser.parse_args()

    video_queue = Queue()
    video_path = args.videopath

    c, fps = video_prepare(video_queue, video_path)
    yolo_predictor = VideoPredictor(video_path, video_queue, num_threads=args.numthreads)

    yolo_predictor.start()
    
    # Засекаем время после запуска потоков
    start = time.monotonic()

    # Ждём пока очередь не станет пустой (все кадры обработаны)
    while not video_queue.empty() or len(yolo_predictor.frames) < c:
        time.sleep(0.1)

    end = time.monotonic()

    # Теперь можно останавливать потоки
    yolo_predictor.stop()

    print(f'Обработка заняла: {end - start:.2f} секунд')
    yolo_predictor.video_decode(fps, c, args.name)

    