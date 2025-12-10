import cv2
import time
import numpy as np
import pytesseract
from ultralytics import YOLO
from rapidfuzz import fuzz, process
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import RPi.GPIO as GPIO
from datetime import datetime

# === 1. CONFIGURATION ===
# Hardware & Network
IP_CAMERA_URL = "http://192.168.1.XX:8080/video"  # ⚠️ แก้ IP กล้องมือถือ
LED_PIN = 17                                      # ขาไฟ LED (จำลองไม้กั้น)

# Paths (อ้างอิงจากไฟล์ Main-GUI.py ของคุณ)
TESSERACT_CMD = "/usr/bin/tesseract"
PLATE_MODEL_PATH = "/home/pi/Desktop/LicensePlate-EdgeAI/LicensePlate.pt"
CODEPROV_MODEL_PATH = "/home/pi/Desktop/LicensePlate-EdgeAI/CodeProv.pt"
PROVINCES_FILE = "/home/pi/Desktop/LicensePlate-EdgeAI/thai_provinces.txt"
CAR_LIST_FILE = "/home/pi/Desktop/LicensePlate-EdgeAI/CarList.txt"

# AI Settings
CONFIDENCE_THRESHOLD = 0.5
LANGUAGE = "tha"

# Google Sheets
SHEET_NAME = "ParkingData"
CREDENTIALS_FILE = "credentials.json"

# === 2. SETUP SYSTEM ===
# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, False)

# Tesseract Setup
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Load Models
print("⏳ Loading AI Models...")
plate_model = YOLO(PLATE_MODEL_PATH)
codeprov_model = YOLO(CODEPROV_MODEL_PATH)
print("✅ Models Loaded")

# Load Text Files
try:
    with open(PROVINCES_FILE, encoding="utf-8") as f:
        thai_provinces = [line.strip() for line in f.readlines()]
    with open(CAR_LIST_FILE, encoding="utf-8") as f:
        car_list = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("❌ Error: ไม่พบไฟล์รายชื่อจังหวัด หรือ CarList")
    thai_provinces = []
    car_list = []

# Connect Google Sheets
print("Connecting to Google Sheets...")
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
try:
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)
    sheet = client.open(SHEET_NAME).sheet1
    print("✅ Google Sheets Connected!")
except Exception as e:
    print(f"❌ Google Sheet Error: {e}")
    sheet = None

# === 3. AI UTILITY FUNCTIONS (จาก Main-GUI.py) ===

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

def plate_detectionandcrop(img):
    # Model 1: หาตำแหน่งป้าย
    results = plate_model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # วาดกรอบสีเขียวที่ภาพต้นฉบับเพื่อความสวยงาม
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return safe_crop(img, x1, y1, x2, y2)
    return None

def seperate_part_and_textOCR(cropped_img):
    # Model 2: แยกเลขทะเบียน กับ จังหวัด
    results = codeprov_model(cropped_img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    code_part, province_part = None, None

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        part_img = safe_crop(cropped_img, x1, y1, x2, y2)

        if cls_id == 0:   # Class 0: เลขทะเบียน
            code_part = part_img
        elif cls_id == 1: # Class 1: จังหวัด
            province_part = part_img

    # เริ่มอ่าน OCR
    code_text = ""
    province_text = ""
    
    # อ่านเลข
    if code_part is not None:
        # Preprocess เล็กน้อย (ขาวดำ) เพื่อให้อ่านแม่นขึ้น
        gray_code = cv2.cvtColor(code_part, cv2.COLOR_BGR2GRAY)
        code_text = pytesseract.image_to_string(gray_code, lang=LANGUAGE, config='--psm 7').strip()
    
    # อ่านจังหวัด
    if province_part is not None:
        gray_prov = cv2.cvtColor(province_part, cv2.COLOR_BGR2GRAY)
        prov_ocr = pytesseract.image_to_string(gray_prov, lang=LANGUAGE, config='--psm 7').strip()
        # ใช้ Fuzzy Logic แก้คำผิด (เช่น 'กรงเทพ' -> 'กรุงเทพมหานคร')
        best_match = process.extractOne(prov_ocr, thai_provinces, scorer=fuzz.ratio)
        if best_match and best_match[1] >= 30: # ความมั่นใจ > 30%
            province_text = best_match[0]

    return code_text, province_text

def open_gate():
    print("🟢 AUTHORIZED: Opening Gate...")
    GPIO.output(LED_PIN, True)
    time.sleep(3) # เปิดค้าง 3 วิ
    GPIO.output(LED_PIN, False)
    print("🔴 Gate Closed")

def log_to_sheets(plate, province, status):
    if sheet:
        try:
            now = datetime.now()
            sheet.append_row([str(now), plate, province, status])
            print(f"📝 Logged to Sheet: {plate} {province}")
        except Exception as e:
            print(f"⚠️ Log Error: {e}")

# === 4. MAIN LOOP ===
cap = cv2.VideoCapture(IP_CAMERA_URL)

# ตัวแปรสำหรับหน่วงเวลา (ไม่ให้ทำงานทุกเสี้ยววินาที เดี๋ยวเครื่องร้อน)
last_process_time = 0
PROCESS_INTERVAL = 1.0  # ตรวจจับทุกๆ 1 วินาที (ปรับลดได้ถ้าอยากให้ไวขึ้น)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read video")
            break

        # ย่อภาพหน่อยเพื่อความเร็วในการแสดงผล
        display_frame = cv2.resize(frame, (640, 480))

        # ตรวจจับตามรอบเวลา (Non-blocking delay)
        current_time = time.time()
        if current_time - last_process_time > PROCESS_INTERVAL:
            
            # 1. ส่งภาพไปหาป้าย (ใช้ภาพต้นฉบับหรือย่อก็ได้ แต่ภาพต้นฉบับแม่นกว่า)
            cropped_plate = plate_detectionandcrop(display_frame) 
            
            if cropped_plate is not None:
                print("🔎 Plate Detected! Analyzing...")
                
                # 2. แยกส่วนและอ่าน OCR
                code_txt, prov_txt = seperate_part_and_textOCR(cropped_plate)
                
                full_plate = code_txt.replace(" ", "")
                full_prov = prov_txt.replace(" ", "")
                
                if full_plate: # ถ้าอ่านเลขออก
                    print(f"🚗 Result: {full_plate} {full_prov}")
                    
                    # 3. เช็คว่าอยู่ในรายชื่อรถ (Whitelist) หรือไม่
                    # (รวมเลขกับจังหวัดเข้าด้วยกันตาม Logic เดิมของคุณ)
                    check_str = full_plate + full_prov 
                    
                    # เช็คแค่เลขทะเบียนใน list ก็ได้ (แก้ logic ตรงนี้ตามไฟล์ text คุณ)
                    is_authorized = False
                    if full_plate in car_list or check_str in car_list:
                        is_authorized = True
                    
                    if is_authorized:
                        log_to_sheets(full_plate, full_prov, "ALLOWED")
                        open_gate() # สั่งเปิดไฟ
                    else:
                        print("⛔ Unknown Car")
                        log_to_sheets(full_plate, full_prov, "UNKNOWN")
                        # กระพริบเตือนว่าไม่ผ่าน
                        GPIO.output(LED_PIN, True); time.sleep(0.2)
                        GPIO.output(LED_PIN, False); time.sleep(0.2)

            last_process_time = current_time

        cv2.imshow("Smart Gate AI", display_frame)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by User")
finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
