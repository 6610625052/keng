import cv2
import time
import numpy as np
import pytesseract
from ultralytics import YOLO
from rapidfuzz import fuzz, process
import RPi.GPIO as GPIO
import threading  # <--- เพิ่ม module นี้เพื่อทำ Multi-threading

# === 1. CONFIGURATION ===
# Hardware
IP_CAMERA_URL = "http://192.168.1.XX:8080/video"  # ⚠️ อย่าลืมแก้ IP
LED_PIN = 17

# Paths
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
try:
    plate_model = YOLO(PLATE_MODEL_PATH)
    codeprov_model = YOLO(CODEPROV_MODEL_PATH)
    print("✅ Models Ready!")
except Exception as e:
    print(f"❌ Error Loading Models: {e}")
    exit()

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

# === 3. CLASS & FUNCTIONS ===

# Class สำหรับอ่านกล้องแบบไม่หน่วง (Threading)
class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print("❌ Cannot open camera source!")
            self.stopped = True
        else:
            (self.grabbed, self.frame) = self.stream.read()
            self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

def plate_detectionandcrop(img):
    results = plate_model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # วาดกรอบสีเขียวที่ป้าย
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return safe_crop(img, x1, y1, x2, y2)
    return None

def seperate_part_and_textOCR(cropped_img):
    results = codeprov_model(cropped_img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    code_part, province_part = None, None

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        part_img = safe_crop(cropped_img, x1, y1, x2, y2)

        if cls_id == 0: code_part = part_img
        elif cls_id == 1: province_part = part_img

    code_text, province_text = "", ""
    
    if code_part is not None:
        gray = cv2.cvtColor(code_part, cv2.COLOR_BGR2GRAY)
        code_text = pytesseract.image_to_string(gray, lang=LANGUAGE, config='--psm 7').strip()
    
    if province_part is not None:
        gray = cv2.cvtColor(province_part, cv2.COLOR_BGR2GRAY)
        txt = pytesseract.image_to_string(gray, lang=LANGUAGE, config='--psm 7').strip()
        match = process.extractOne(txt, thai_provinces, scorer=fuzz.ratio)
        if match and match[1] >= 30: province_text = match[0]

    return code_text, province_text

# ฟังก์ชันเปิดประตูแบบไม่บล็อกการทำงาน (แยก Thread)
def activate_gate_thread():
    def run():
        print("🟢 LED ON (Gate Open)")
        GPIO.output(LED_PIN, True)
        time.sleep(3) 
        GPIO.output(LED_PIN, False)
        print("🔴 LED OFF (Gate Closed)")
    
    # สั่งให้ทำงานแยกออกไปเลย วิดีโอจะได้ไม่ค้าง
    threading.Thread(target=run, daemon=True).start()

def blink_warning():
    def run():
        for _ in range(3):
            GPIO.output(LED_PIN, True)
            time.sleep(0.1)
            GPIO.output(LED_PIN, False)
            time.sleep(0.1)
    threading.Thread(target=run, daemon=True).start()

# === 4. MAIN LOOP ===
print("🚀 Starting Camera Stream...")
# ใช้ Class ใหม่แทน cv2.VideoCapture โดยตรง
cam = CameraStream(IP_CAMERA_URL).start()
time.sleep(1.0) # รอให้กล้องเริ่มทำงานนิดนึง

last_process = 0
frame_count = 0

print("Press 'q' to exit")

try:
    while True:
        frame = cam.read()
        
        if frame is None:
            continue # ถ้าไม่มีภาพ ให้ข้ามรอบนี้ไป
        
        # ย่อภาพก่อนเลย เพื่อลดภาระเครื่อง
        display_frame = cv2.resize(frame, (640, 480))
        
        # AI ทำงานทุกๆ 1 วินาที
        if time.time() - last_process > 1.0:
            
            # ใช้ display_frame ที่ย่อแล้วส่งเข้า AI จะเร็วกว่า
            crop = plate_detectionandcrop(display_frame)
            
            if crop is not None:
                c_txt, p_txt = seperate_part_and_textOCR(crop)
                cv2.imshow("Cropped Plate", crop)
                full_plate = c_txt.replace(" ", "")
                full_prov = p_txt.replace(" ", "")
                
                if full_plate:
                    print(f"👀 Saw: {full_plate} {full_prov}")
                    
                    check_str = full_plate + full_prov
                    # เช็คว่ามีอยู่ในรายการไหม
                    is_authorized = False
                    for car in car_list:
                        # เช็คแบบบ้านๆ (ถ้าข้อความที่อ่านได้ ไปอยู่ใน whitelist ส่วนใดส่วนหนึ่ง)
                        if full_plate in car or check_str in car:
                            is_authorized = True
                            break
                    
                    if is_authorized:
                        print(f"✅ PASS: Authorized ({full_plate})")
                        activate_gate_thread() # เรียกแบบใหม่ ไม่ค้าง
                    else:
                        print(f"⛔ STOP: Unknown Car ({full_plate})")
                        blink_warning() # เรียกแบบใหม่ ไม่ค้าง
            
            last_process = time.time()
            
        cv2.imshow("Smart Gate System", display_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by User")
finally:
    cam.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()
