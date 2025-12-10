import cv2
import pytesseract
import time
from ultralytics import YOLO
import numpy as np
from rapidfuzz import fuzz, process

# === CONFIG ===
TESSERACT_CMD = "/usr/bin/tesseract"
LANGUAGE = "tha+eng"
PLATE_MODEL_PATH = "/home/cn360/Desktop/LicensePlate-EdgeAI/LicensePlate.pt"
CODEPROV_MODEL_PATH = "/home/cn360/Desktop/LicensePlate-EdgeAI/CodeProv.pt"
CONFIDENCE_THRESHOLD = 0.5
IP_CAM_URL = "http://10.72.93.71:8080/video" # <--- ใส่ URL ของคุณที่นี่

# === INIT ===
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
print("Loading Models...")
plate_model = YOLO(PLATE_MODEL_PATH)
codeprov_model = YOLO(CODEPROV_MODEL_PATH)
print("Models Loaded.")

with open("/home/cn360/Desktop/LicensePlate-EdgeAI/thai_provinces.txt", encoding="utf-8") as f:
    thai_provinces = [line.strip() for line in f.readlines()]

# === UTILS (เหมือนเดิม) ===
def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def match_province(input_text, threshold=75):
    best_match = process.extractOne(input_text, thai_provinces, scorer=fuzz.ratio)
    if best_match and best_match[1] >= threshold:
        return best_match[0]
    return None

# === CORE FUNCTIONS ===
def process_frame(img):
    # 1. Detect Plate
    plate_results = plate_model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    
    for box in plate_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped_plate = safe_crop(img, x1, y1, x2, y2)
        
        # วาดกรอบสี่เหลี่ยมสีเขียวบนภาพจริงเพื่อให้รู้ว่าจับเจอ
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 2. Detect Text inside Plate
        ocr_results = codeprov_model(cropped_plate, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        code_part, province_part = None, None

        for ocr_box in ocr_results.boxes:
            px1, py1, px2, py2 = map(int, ocr_box.xyxy[0].tolist())
            cls_id = int(ocr_box.cls[0])
            part_img = safe_crop(cropped_plate, px1, py1, px2, py2)
            
            if cls_id == 0: # Code
                code_part = preprocess_for_ocr(part_img)
            elif cls_id == 1: # Province
                province_part = preprocess_for_ocr(part_img)

        # 3. OCR
        code_text = pytesseract.image_to_string(code_part, lang=LANGUAGE, config='--psm 7').strip() if code_part is not None else ''
        province_raw = pytesseract.image_to_string(province_part, lang=LANGUAGE, config='--psm 7').strip() if province_part is not None else ''
        
        province_clean = match_province(province_raw)
        final_province = province_clean if province_clean else province_raw

        return f"{code_text} {final_province}", img
    
    return None, img

# === MAIN LOOP ===
def main():
    print(f"Connecting to Camera: {IP_CAM_URL}")
    cap = cv2.VideoCapture(IP_CAM_URL)

    # ตั้งค่าขนาดภาพเพื่อลดภาระเครื่อง (Optional)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    last_detected_text = ""
    last_detected_time = 0
    COOLDOWN_SECONDS = 5.0  # เวลาหน่วง ถ้าเจอทะเบียนเดิมจะไม่ Print ซ้ำภายใน 5 วินาที

    print("Starting Loop. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            # ถ้ากล้องหลุด ให้ลองเชื่อมต่อใหม่ หรือ break
            time.sleep(1) 
            continue

        # ส่งภาพไปประมวลผล
        result_text, display_frame = process_frame(frame)

        if result_text:
            current_time = time.time()
            # เช็คว่าทะเบียนซ้ำกับเมื่อกี้ไหม และเวลาผ่านไปนานพอหรือยัง
            if result_text != last_detected_text or (current_time - last_detected_time > COOLDOWN_SECONDS):
                
                print(f"📌 DETECTED: {result_text}") # <--- Output ตรงนี้
                
                last_detected_text = result_text
                last_detected_time = current_time
                
                # บันทึกภาพเมื่อเจอ
                cv2.imwrite("last_detected.jpg", display_frame)

        # แสดงภาพสด (Live View)
        cv2.imshow("LPR System (Press 'q' to exit)", display_frame)

        # กด 'q' เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
