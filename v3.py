import cv2
import time
import numpy as np
import requests
import threading
import queue
import pytesseract
from ultralytics import YOLO
from rapidfuzz import fuzz, process
import RPi.GPIO as GPIO

# === 1. CONFIGURATION ===
IP_CAMERA_URL = "http://10.72.93.71:8080/shot.jpg"  # ⚠️ ใช้ /shot.jpg เพื่อความลื่น
LED_PIN = 17

# Paths
TESSERACT_CMD = "/usr/bin/tesseract"
PLATE_MODEL_PATH = "/home/cn360/Desktop/LicensePlate-EdgeAI/LicensePlate.pt"
CODEPROV_MODEL_PATH = "/home/cn360/Desktop/LicensePlate-EdgeAI/CodeProv.pt"
PROVINCES_FILE = "/home/cn360/Desktop/LicensePlate-EdgeAI/thai_provinces.txt"
CAR_LIST_FILE = "/home/cn360/Desktop/LicensePlate-EdgeAI/Carlist.txt"
CONFIDENCE_THRESHOLD = 0.5
LANGUAGE = "tha"

# Global Variables สำหรับส่งข้อมูลข้าม Thread
latest_frame = None  # เก็บภาพล่าสุด
detection_result = None  # เก็บผลลัพธ์ล่าสุดจาก AI (กรอบ, ชื่อทะเบียน)
frame_lock = threading.Lock()
running = True

# === 2. SETUP ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, False)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

print("⏳ Loading AI Models (Background)...")
# โหลด Model เตรียมไว้เลย
plate_model = YOLO(PLATE_MODEL_PATH)
codeprov_model = YOLO(CODEPROV_MODEL_PATH)

# Load Lists
try:
    with open(PROVINCES_FILE, encoding="utf-8") as f:
        thai_provinces = [line.strip() for line in f.readlines()]
    with open(CAR_LIST_FILE, encoding="utf-8") as f:
        car_list = [line.strip() for line in f.readlines()]
except Exception:
    thai_provinces, car_list = [], []
print("✅ Ready!")

# === 3. AI WORKER THREAD ===
def ai_worker():
    global latest_frame, detection_result, running
    
    print("🤖 AI Brain Started...")
    while running:
        img_to_process = None
        
        # 1. แอบไปจิ๊กภาพล่าสุดมา (ถ้ามี)
        with frame_lock:
            if latest_frame is not None:
                img_to_process = latest_frame.copy()
        
        if img_to_process is None:
            time.sleep(0.1)
            continue

        # 2. เริ่มประมวลผล (กินเวลา 0.X วินาที แต่ไม่กระทบหน้าจอหลัก)
        # --- (Logic เดิมของคุณ) ---
        # หาป้าย
        results = plate_model(img_to_process, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        found_plate = False
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # ตัดภาพป้าย
            h, w = img_to_process.shape[:2]
            crop = img_to_process[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            # แยกเลข/จังหวัด
            res_ocr = codeprov_model(crop, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            code_part, prov_part = None, None
            
            for b in res_ocr.boxes:
                cx1, cy1, cx2, cy2 = map(int, b.xyxy[0].tolist())
                cid = int(b.cls[0])
                p_img = crop[max(0, cy1):min(crop.shape[0], cy2), max(0, cx1):min(crop.shape[1], cx2)]
                if cid == 0: code_part = p_img
                elif cid == 1: prov_part = p_img
            
            # OCR
            full_text = ""
            if code_part is not None:
                gray = cv2.cvtColor(code_part, cv2.COLOR_BGR2GRAY)
                full_text += pytesseract.image_to_string(gray, lang=LANGUAGE, config='--psm 7').strip()
            
            if prov_part is not None:
                gray = cv2.cvtColor(prov_part, cv2.COLOR_BGR2GRAY)
                txt = pytesseract.image_to_string(gray, lang=LANGUAGE, config='--psm 7').strip()
                match = process.extractOne(txt, thai_provinces, scorer=fuzz.ratio)
                if match and match[1] >= 30: full_text += " " + match[0]
            
            clean_text = full_text.replace(" ", "")
            
            # ตรวจสอบสิทธิ์
            is_authorized = False
            for c in car_list:
                if clean_text in c or c in clean_text:
                    is_authorized = True
                    break

            # อัปเดตผลลัพธ์กลับไปให้หน้าจอหลัก
            detection_result = {
                "box": (x1, y1, x2, y2),
                "text": full_text,
                "auth": is_authorized
            }
            
            # ถ้าผ่าน ให้สั่งเปิดประตู (แยก Thread อีกทีกันเหนียว)
            if is_authorized:
                threading.Thread(target=open_gate_action, daemon=True).start()
            elif clean_text:
                threading.Thread(target=blink_action, daemon=True).start()
                
            found_plate = True
            break # เจอแล้วหยุดหา
            
        if not found_plate:
            detection_result = None # ไม่เจออะไรก็เคลียร์ผลลัพธ์ทิ้ง

        # พักนิดนึง ไม่ให้ CPU ร้อนจัด (ปรับได้)
        time.sleep(0.01) 

# === 4. HARDWARE ACTIONS ===
def open_gate_action():
    print("🟢 GATE OPEN")
    GPIO.output(LED_PIN, True)
    time.sleep(3)
    GPIO.output(LED_PIN, False)

def blink_action():
    for _ in range(3):
        GPIO.output(LED_PIN, True); time.sleep(0.1)
        GPIO.output(LED_PIN, False); time.sleep(0.1)

# === 5. MAIN LOOP (DISPLAY) ===
# เริ่ม AI Thread
ai_thread = threading.Thread(target=ai_worker, daemon=True)
ai_thread.start()

print("🚀 Starting Display...")
cap_url = IP_CAMERA_URL

try:
    while True:
        # 1. ดึงภาพ (Snapshot)
        try:
            resp = requests.get(cap_url, timeout=2)
            arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = cv2.imdecode(arr, -1)
            
            if frame is not None:
                # ย่อภาพให้แสดงผลลื่น
                frame = cv2.resize(frame, (640, 480))
                
                # ส่งภาพไปให้ AI (Update Global Variable)
                with frame_lock:
                    latest_frame = frame.copy()
                
                # 2. วาดผลลัพธ์จาก AI (ถ้ามี)
                # เราใช้ข้อมูลเดิมวาดซ้ำไปเรื่อยๆ จนกว่า AI จะส่งข้อมูลใหม่มา (ทำให้ตากระพริบน้อยลง)
                if detection_result:
                    x1, y1, x2, y2 = detection_result["box"]
                    text = detection_result["text"]
                    color = (0, 255, 0) if detection_result["auth"] else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.imshow("Smart Gate (Async)", frame)
            
        except Exception as e:
            print(f"Cam Error: {e}")
            time.sleep(1)

        if cv2.waitKey(1) == ord('q'):
            running = False
            break

except KeyboardInterrupt:
    pass
finally:
    running = False
    cv2.destroyAllWindows()
    GPIO.cleanup()
