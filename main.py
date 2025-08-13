import cv2
import threading
import time
from collections import deque

def list_available_cameras(max_tested=5):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

class CameraThread(threading.Thread):
    def __init__(self, cam_id, buffer_size=10, width=640, height=480, fps=15):
        super().__init__()
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)
        self.buffer = deque(maxlen=buffer_size)
        self.running = True
        
        # Safe resolution & FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
    
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            timestamp = time.time()
            self.buffer.append((timestamp, frame))
    
    def stop(self):
        self.running = False
        self.cap.release()

def synchronize_frames(cams):
    master_ts, master_frame = cams[0].buffer[-1]
    synced_frames = [master_frame]
    for cam in cams[1:]:
        closest = min(cam.buffer, key=lambda x: abs(x[0] - master_ts))
        synced_frames.append(closest[1])
    return synced_frames

if __name__ == "__main__":
    cam_ids = list_available_cameras()
    print(f"Detected cameras: {cam_ids}")
    
    if len(cam_ids) < 2:
        print("❌ Need at least 2 cameras connected!")
        exit()

    cameras = [CameraThread(cid) for cid in cam_ids]
    for cam in cameras:
        cam.start()
    
    writers = [
        cv2.VideoWriter(f"cam_{i}.avi", cv2.VideoWriter_fourcc(*"XVID"), 15, (640, 480))
        for i in range(len(cameras))
    ]
    
     # Wait until all cameras have at least 1 frame
    print("Warming up cameras...")
    while any(len(cam.buffer) == 0 for cam in cameras):
        time.sleep(0.1)

    try:
        while True:
            frames = synchronize_frames(cameras)
            for i, frame in enumerate(frames):
                cv2.imshow(f"Camera {i}", frame)
                writers[i].write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for cam in cameras:
            cam.stop()
        for cam in cameras:
            cam.join()
        for w in writers:
            w.release()
        cv2.destroyAllWindows()
