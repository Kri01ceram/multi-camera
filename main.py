import cv2
import threading
import time
import os
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
    # Load Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"⚠️ Could not load Haar cascade at {cascade_path}. Face detection will be disabled.")

    # Optional directory to save detected face crops
    faces_dir = os.path.join(os.getcwd(), "detected_faces")
    os.makedirs(faces_dir, exist_ok=True)

    # Configure saving behavior:
    # If SAVE_FACE_ONCE is False, no face crops are written to disk.
    # If True, each unique face (per camera) is saved once using centroid-distance deduplication.
    SAVE_FACE_ONCE = False
    # Minimum pixel distance between centroids to consider a face "new" (tweak as needed)
    MIN_CENTROID_DIST = 60
    # Track saved face centroids per camera
    saved_face_centroids = [[] for _ in range(len(cameras))]
    
     # Wait until all cameras have at least 1 frame
    print("Warming up cameras...")
    while any(len(cam.buffer) == 0 for cam in cameras):
        time.sleep(0.1)

    try:
        while True:
            frames = synchronize_frames(cameras)
            for i, frame in enumerate(frames):
                # Detect faces and annotate the frame
                if not face_cascade.empty():
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    for idx, (x, y, w, h) in enumerate(faces):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Face {idx+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        # Save small crops of detected faces only when enabled.
                        if SAVE_FACE_ONCE:
                            # Compute centroid of the detected face
                            cx = x + w / 2
                            cy = y + h / 2
                            is_new = True
                            for (sx, sy) in saved_face_centroids[i]:
                                dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
                                if dist < MIN_CENTROID_DIST:
                                    is_new = False
                                    break
                            if is_new:
                                try:
                                    crop = frame[y:y+h, x:x+w]
                                    ts = int(time.time() * 1000)
                                    fname = os.path.join(faces_dir, f"cam{i}_face{idx+1}_{ts}.jpg")
                                    cv2.imwrite(fname, crop)
                                    saved_face_centroids[i].append((cx, cy))
                                except Exception:
                                    # If saving fails, continue without interrupting capture
                                    pass

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
