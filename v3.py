import cv2
import time
import numpy as np
import requests
import threading
import pytesseract
from ultralytics import YOLO
from rapidfuzz import fuzz, process
import RPi.GPIO as GPIO

# === 1. CONFIGURATION ===
IP_CAMERA_URL = "http://10.72.93.71:8080/shot.jpg"  # ⚠️ เช็ค IP อีกทีนะครับ
LED_PIN = 17

# Paths
TESSERACT_CMD = "/usr/bin/tesseract"
PLATE_MODEL_PATH = "/home/cn360/Desktop/LicensePlate-EdgeAI/LicensePlate.pt"
CODEPROV_MODEL_PATH = "/home/cn360/Desktop/LicensePlate-EdgeAI/CodeProv.pt"
PROVINCES_FILE = "/home/cn360/Desktop/LicensePlate-EdgeAI/thai_provinces.txt"
CAR_LIST_FILE = "/home/cn360/Desktop/LicensePlate-EdgeAI/Carlist.txt"

CONFIDENCE_THRESHOLD = 0.5
LANGUAGE = "tha"
THAI_CHARS = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
NUMBERS = "0123456789"

# Global Variables
latest_frame = None
detection_result = None
frame_lock = threading.Lock()
running = True

# === 2. SETUP ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, False)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

print("⏳ Loading AI Models (Background)...")
plate_model = YOLO(PLATE_MODEL_PATH)
codeprov_model = YOLO(CODEPROV_MODEL_PATH)

try:
    with open(PROVINCES_FILE, encoding="utf-8") as f:
        thai_provinces = [line.strip() for line in f.readlines()]
    with open(CAR_LIST_FILE, encoding="utf-8") as f:
        car_list = [line.strip() for line in f.readlines()]
except Exception:
    thai_provinces, car_list = [], []
print("✅ Ready!")

# === 3. HELPER FUNCTIONS (เหมือน Main-GUI) ===
def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

def preprocess_for_ocr(img):
    # แปลงขาวดำ + ขยายภาพ + ตัด Noise (สูตรสำเร็จให้อ่านแม่น)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def match_province(input_text, threshold=30):
    best_match = process.extractOne(input_text, thai_provinces, scorer=fuzz.ratio)
    if best_match and best_match[1] >= threshold:
        return best_match[0]
    return ""

# === 4. AI WORKER THREAD ===
def ai_worker():
    global latest_frame, detection_result, running
    
    print("🤖 AI Brain Started...")
    while running:
        img_to_process = None
        
        with frame_lock:
            if latest_frame is not None:
                img_to_process = latest_frame.copy()
        
        if img_to_process is None:
            time.sleep(0.1)
            continue

        # เริ่มตรวจจับ
        results = plate_model(img_to_process, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        found_plate = False
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Crop ป้ายทะเบียน
            crop = safe_crop(img_to_process, x1, y1, x2, y2)
            
            # แยกเลข / จังหวัด
            res_ocr = codeprov_model(crop, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            code_part, prov_part = None, None
            
            for b in res_ocr.boxes:
                cx1, cy1, cx2, cy2 = map(int, b.xyxy[0].tolist())
                cid = int(b.cls[0])
                p_img = safe_crop(crop, cx1, cy1, cx2, cy2)
                
                if cid == 0: code_part = p_img
                elif cid == 1: prov_part = p_img
            
            # OCR Reading
            code_text = ""
            province_text = ""

            # อ่านเลขทะเบียน (บังคับ whitelist ไทย+เลข)
            if code_part is not None:
                processed_code = preprocess_for_ocr(code_part)
                custom_config = f'--psm 7 -c tessedit_char_whitelist={THAI_CHARS}{NUMBERS}'
                code_text = pytesseract.image_to_string(processed_code, lang=LANGUAGE, config=custom_config).strip()
            
            # อ่านจังหวัด (บังคับ whitelist ไทยอย่างเดียว)
            if prov_part is not None:
                processed_prov = preprocess_for_ocr(prov_part)
                custom_config = f'--psm 7 -c tessedit_char_whitelist={THAI_CHARS}'
                raw_prov = pytesseract.image_to_string(processed_prov, lang=LANGUAGE, config=custom_config).strip()
                province_text = match_province(raw_prov)
            
            full_text_display = f"{code_text} {province_text}"
            clean_text_check = code_text.replace(" ", "") + province_text.replace(" ", "")
            
            # ตรวจสอบสิทธิ์ (Whitelist Check)
            is_authorized = False
            for c in car_list:
                # เช็คว่าข้อความที่อ่านได้ มีอยู่ในรายการรถไหม
                if clean_text_check and (clean_text_check in c or c in clean_text_check):
                    is_authorized = True
                    break

            # ส่งผลลัพธ์กลับไปหน้าจอหลัก
            detection_result = {
                "box": (x1, y1, x2, y2),
                "text": full_text_display,
                "auth": is_authorized
            }
            
            # สั่ง Hardware
            if is_authorized:
                threading.Thread(target=open_gate_action, daemon=True).start()
            elif clean_text_check:
                threading.Thread(target=blink_action, daemon=True).start()
                
            found_plate = True
            break 
            
        if not found_plate:
            detection_result = None

        time.sleep(0.01) 

# === 5. HARDWARE ACTIONS ===
def open_gate_action():
    print("🟢 GATE OPEN")
    GPIO.output(LED_PIN, True)
    time.sleep(3)
    GPIO.output(LED_PIN, False)

def blink_action():
    for _ in range(3):
        GPIO.output(LED_PIN, True); time.sleep(0.1)
        GPIO.output(LED_PIN, False); time.sleep(0.1)

# === 6. MAIN LOOP ===
ai_thread = threading.Thread(target=ai_worker, daemon=True)
ai_thread.start()

print("🚀 Starting Display...")
cap_url = IP_CAMERA_URL

try:
    while True:
        try:
            # ดึงภาพแบบ Snapshot (เร็วสุดสำหรับ IP Cam)
            resp = requests.get(cap_url, timeout=2)
            arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = cv2.imdecode(arr, -1)
            
            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
                
                # ส่งภาพล่าสุดให้ AI
                with frame_lock:
                    latest_frame = frame.copy()
                
                # วาดผลลัพธ์
                if detection_result:
                    x1, y1, x2, y2 = detection_result["box"]
                    text = detection_result["text"]
                    color = (0, 255, 0) if detection_result["auth"] else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # ใส่พื้นหลังตัวหนังสือให้อ่านง่าย
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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
