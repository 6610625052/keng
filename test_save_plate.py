import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

# === CONFIGURATION ===
# 1. ตั้งค่ากล้อง IP Camera
IP_CAM_URL = "http://10.72.93.71:8080/video" 

# 2. ตั้งค่าโฟลเดอร์ที่จะบันทึกรูป
SAVE_FOLDER = "captured_plates"

# 3. Path ของ Model
PLATE_MODEL_PATH = "/home/cn360/Desktop/LicensePlate-EdgeAI/LicensePlate.pt"
CONFIDENCE_THRESHOLD = 0.6

# 4. ตั้งค่ากันบันทึกซ้ำ (Cooldown)
SAVE_COOLDOWN = 5.0 # วินาที (ถ้าเจอรถคันเดิม จะไม่บันทึกซ้ำภายในเวลานี้)

# === INIT ===
print("Loading YOLO Model...")
plate_model = YOLO(PLATE_MODEL_PATH)
print("Model Loaded.")

# สร้างโฟลเดอร์ถ้ายังไม่มี
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
    print(f"Created folder: {SAVE_FOLDER}")

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

# === MAIN PROCESS ===
def main():
    print(f"Connecting to IP Camera: {IP_CAM_URL}")
    cap = cv2.VideoCapture(IP_CAM_URL)
    
    # ลดขนาดภาพลงเพื่อความลื่น (Optional)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_saved_time = 0

    print(f"🟢 System Ready. Saving images to folder '{SAVE_FOLDER}'")
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame (reconnecting...)")
            time.sleep(1)
            cap = cv2.VideoCapture(IP_CAM_URL)
            continue

        # 1. ให้ YOLO หาป้ายทะเบียน
        results = plate_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        
        detected = False
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # วาดกรอบสีเขียวบนหน้าจอ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ตรวจสอบเวลา (Cooldown) ว่าควรบันทึกหรือไม่
            current_time = time.time()
            if (current_time - last_saved_time) > SAVE_COOLDOWN:
                
                # Crop ภาพเฉพาะป้าย
                cropped_plate = safe_crop(frame, x1, y1, x2, y2)
                
                # สร้างชื่อไฟล์จากวันเวลา (เช่น plate_2023-12-09_14-30-01.jpg)
                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{SAVE_FOLDER}/plate_{timestamp_str}.jpg"
                
                # บันทึกลงเครื่อง
                cv2.imwrite(filename, cropped_plate)
                print(f"💾 Saved: {filename}")
                
                last_saved_time = current_time
                detected = True
                
            # ถ้าเจอหลายป้าย เอาแค่ป้ายแรกพอ (break loop นี้)
            if detected:
                break

        # แสดงผลหน้าจอ
        cv2.imshow("Test Capture Local", frame)
        
        # กด q เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
