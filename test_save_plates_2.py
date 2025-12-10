import cv2
import time
import os
import threading
from datetime import datetime
from ultralytics import YOLO

# === CONFIGURATION ===
# เช็ค URL ให้ชัวร์ว่าลงท้ายด้วย /video หรือเป็น stream ไม่ใช่ .jpg
IP_CAM_URL = "http://10.72.93.71:8080/video" 

SAVE_FOLDER = "captured_plates"
PLATE_MODEL_PATH = "/home/cn360/Desktop/LicensePlate-EdgeAI/LicensePlate.pt"
CONFIDENCE_THRESHOLD = 0.6
SAVE_COOLDOWN = 5.0

# === CLASS: แยกการทำงานกล้องออกมาเป็นอีก Thread ===
class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # ปรับความละเอียดลงเพื่อให้ลื่นขึ้น (ถ้ากล้องรองรับ)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # เริ่ม Thread แยก
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # ลูปอ่านภาพทิ้งไปเรื่อยๆ เพื่อเคลียร์ Buffer ให้ว่าง และเก็บภาพล่าสุดไว้
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # คืนค่าภาพล่าสุด
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# === UTILS ===
def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

# === MAIN ===
def main():
    print("Loading YOLO Model...")
    plate_model = YOLO(PLATE_MODEL_PATH)
    
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    print(f"Connecting to Camera: {IP_CAM_URL}")
    
    # เรียกใช้คลาส CameraStream แทน cv2.VideoCapture ปกติ
    cam = CameraStream(IP_CAM_URL).start()
    time.sleep(1.0) # รอให้กล้องวอร์มเครื่องแป๊บนึง

    last_saved_time = 0

    print("🟢 System Ready. Press 'q' to exit.")

    while True:
        # อ่านภาพจาก Thread พิเศษ
        frame = cam.read()
        
        # ถ้าไม่มีภาพ (กล้องหลุด) ให้ข้ามไป
        if frame is None:
            continue

        # --- ส่วน AI (เหมือนเดิม) ---
        results = plate_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detected = False
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # วาดกรอบ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            current_time = time.time()
            if (current_time - last_saved_time) > SAVE_COOLDOWN:
                cropped_plate = safe_crop(frame, x1, y1, x2, y2)
                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{SAVE_FOLDER}/plate_{timestamp_str}.jpg"
                
                cv2.imwrite(filename, cropped_plate)
                print(f"💾 Saved: {filename}")
                
                last_saved_time = current_time
                detected = True
            
            if detected:
                break

        # แสดงภาพ
        cv2.imshow("Smoother Stream", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.stop() # อย่าลืมสั่งหยุดกล้อง
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
