import cv2
import time
import numpy as np
import pytesseract
from ultralytics import YOLO
from rapidfuzz import fuzz, process
import RPi.GPIO as GPIO

# === 1. CONFIGURATION ===
# Hardware
IP_CAMERA_URL = "http://192.168.1.XX:8080/video"  # ⚠️ แก้ IP มือถือให้ตรง
LED_PIN = 17

# Paths (เช็ค path ให้ตรงกับเครื่องจริง)
TESSERACT_CMD = "/usr/bin/tesseract"
PLATE_MODEL_PATH = "/home/pi/Desktop/LicensePlate-EdgeAI/LicensePlate.pt"
CODEPROV_MODEL_PATH = "/home/pi/Desktop/LicensePlate-EdgeAI/CodeProv.pt"
PROVINCES_FILE = "/home/pi/Desktop/LicensePlate-EdgeAI/thai_provinces.txt"
CAR_LIST_FILE = "/home/pi/Desktop/LicensePlate-EdgeAI/CarList.txt"

# AI Settings
CONFIDENCE_THRESHOLD = 0.5
LANGUAGE = "tha"

# === 2. SETUP ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, False)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

print("⏳ Loading AI Models...")
plate_model = YOLO(PLATE_MODEL_PATH)
codeprov_model = YOLO(CODEPROV_MODEL_PATH)
print("✅ Models Ready!")

# Load Lists
try:
    with open(PROVINCES_FILE, encoding="utf-8") as f:
        thai_provinces = [line.strip() for line in f.readlines()]
    with open(CAR_LIST_FILE, encoding="utf-8") as f:
        car_list = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(car_list)} cars in whitelist.")
except FileNotFoundError:
    print("❌ Error: Missing text files")
    thai_provinces, car_list = [], []

# === 3. FUNCTIONS ===
def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

def plate_detectionandcrop(img):
    # Model 1: หาตำแหน่งป้าย
    results = plate_model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return safe_crop(img, x1, y1, x2, y2)
    return None

def seperate_part_and_textOCR(cropped_img):
    # Model 2: แยกเลขกับจังหวัด
    results = codeprov_model(cropped_img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    code_part, province_part = None, None

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        part_img = safe_crop(cropped_img, x1, y1, x2, y2)

        if cls_id == 0: code_part = part_img
        elif cls_id == 1: province_part = part_img

    code_text = ""
    province_text = ""
    
    # OCR อ่านเลข
    if code_part is not None:
        gray = cv2.cvtColor(code_part, cv2.COLOR_BGR2GRAY)
        code_text = pytesseract.image_to_string(gray, lang=LANGUAGE, config='--psm 7').strip()
    
    # OCR อ่านจังหวัด
    if province_part is not None:
        gray = cv2.cvtColor(province_part, cv2.COLOR_BGR2GRAY)
        txt = pytesseract.image_to_string(gray, lang=LANGUAGE, config='--psm 7').strip()
        match = process.extractOne(txt, thai_provinces, scorer=fuzz.ratio)
        if match and match[1] >= 30: province_text = match[0]

    return code_text, province_text

def open_gate():
    print("🟢 LED ON (Gate Open)")
    GPIO.output(LED_PIN, True)
    time.sleep(3) # เปิดไฟค้าง 3 วิ
    GPIO.output(LED_PIN, False)
    print("🔴 LED OFF (Gate Closed)")

# === 4. MAIN LOOP ===
cap = cv2.VideoCapture(IP_CAMERA_URL)
last_process = 0

print("🚀 Starting Camera... Press 'q' to exit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read video - Check IP Camera")
            break
        
        # ย่อภาพให้แสดงผลลื่นๆ
        display_frame = cv2.resize(frame, (640, 480))
        
        # ตรวจจับทุกๆ 1 วินาที (ไม่ให้เครื่องร้อนเกินไป)
        if time.time() - last_process > 1.0:
            
            # 1. หาป้าย
            crop = plate_detectionandcrop(display_frame)
            
            if crop is not None:
                # 2. อ่านป้าย
                c_txt, p_txt = seperate_part_and_textOCR(crop)
                full_plate = c_txt.replace(" ", "")
                full_prov = p_txt.replace(" ", "")
                
                if full_plate:
                    print(f"👀 Saw: {full_plate} {full_prov}")
                    
                    # 3. เช็คเงื่อนไข
                    check_str = full_plate + full_prov
                    if full_plate in car_list or check_str in car_list:
                        print("✅ PASS: Authorized")
                        open_gate()
                    else:
                        print("⛔ STOP: Unknown Car")
                        # กระพริบไฟเตือน
                        GPIO.output(LED_PIN, True); time.sleep(0.1)
                        GPIO.output(LED_PIN, False); time.sleep(0.1)
            
            last_process = time.time()
            
        cv2.imshow("Test Camera Mode", display_frame)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by User")
finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
