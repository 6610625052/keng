import cv2
import time
import requests
import threading
from ultralytics import YOLO
import numpy as np

# === CONFIGURATION ===
# 1. ตั้งค่ากล้อง IP Camera
IP_CAM_URL = "http://10.72.93.71:8080/video" 

# 2. ตั้งค่า n8n Webhook URL (POST)
N8N_WEBHOOK_URL = "https://chonchanan.app.n8n.cloud/webhook-test/888"

# 3. Path Model
PLATE_MODEL_PATH = "/home/cn360/Desktop/LicensePlate-EdgeAI/LicensePlate.pt"
CONFIDENCE_THRESHOLD = 0.6

# 4. ตั้งค่ากันส่งซ้ำ (Cooldown)
SEND_COOLDOWN = 5.0 

# === CLASS: อ่านภาพจากกล้องแบบแยก Thread (แก้ภาพค้าง) ===
class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # ปรับความละเอียดลงเพื่อให้ลื่นขึ้น (ถ้าจำเป็น)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# === UTILS ===
def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

def upload_worker(image_array, url):
    """ฟังก์ชันสำหรับอัปโหลดรูป (รันใน Thread แยก)"""
    try:
        # แปลงภาพเป็นไฟล์ jpg ใน memory
        _, img_encoded = cv2.imencode('.jpg', image_array)
        files = {'file': ('plate.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        # ส่งข้อมูล
        # print("🚀 Sending to n8n...") 
        response = requests.post(url, files=files, timeout=10)
        
        if response.status_code == 200:
            print("✅ Upload Success")
        else:
            print(f"❌ Upload Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Upload Error: {e}")

# === MAIN PROCESS ===
def main():
    print("Loading YOLO Model...")
    plate_model = YOLO(PLATE_MODEL_PATH)
    
    print(f"Connecting to Camera: {IP_CAM_URL}")
    cam = CameraStream(IP_CAM_URL).start()
    time.sleep(1.0) # รอเชื่อมต่อ

    last_sent_time = 0

    print("🟢 System Ready. Press 'q' to exit.")

    while True:
        # 1. รับภาพจาก Thread กล้อง
        frame = cam.read()
        
        if frame is None:
            continue

        # 2. ให้ YOLO หาป้ายทะเบียน
        # ใช้ verbose=False เพื่อลดข้อความรกใน Terminal
        results = plate_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        
        detected = False
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # วาดกรอบสีเขียว
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # เช็คเวลา Cooldown
            current_time = time.time()
            if (current_time - last_sent_time) > SEND_COOLDOWN:
                
                # Crop ภาพ
                cropped_plate = safe_crop(frame, x1, y1, x2, y2)
                
                # *** จุดสำคัญ: สั่งให้ส่งรูปใน "Thread ใหม่" ***
                # เพื่อไม่ให้ Main Loop ต้องรออัปโหลดเสร็จ (ซึ่งจะทำให้จอค้าง)
                sender_thread = threading.Thread(target=upload_worker, args=(cropped_plate.copy(), N8N_WEBHOOK_URL))
                sender_thread.start()
                
                print(f"🚀 Detected! Sending in background...")
                
                last_sent_time = current_time
                detected = True
            
            if detected:
                break

        # 3. แสดงผล (ส่วนนี้จะลื่น เพราะไม่มีอะไรมาขวางการทำงาน)
        cv2.imshow("LPR to n8n (Smoothed)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.stop()
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
