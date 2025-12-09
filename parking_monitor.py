import paho.mqtt.client as mqtt

# ตั้งค่า
TOTAL_SLOTS = 10
current_cars = 0

# เมื่อเชื่อมต่อสำเร็จ
def on_connect(client, userdata, flags, rc):
    print("Connected to Broker!")
    # บอกว่าจะรอฟังข่าวจากหัวข้อ parking/slot1
    client.subscribe("parking/slot1")

# เมื่อมีข้อความเข้ามา
def on_message(client, userdata, msg):
    global current_cars
    payload = msg.payload.decode()
    print(f"Update from {msg.topic}: {payload}")

    if payload == "OCCUPIED":
        current_cars += 1
        print(f"--> รถเข้าจอด! (รถทั้งหมด: {current_cars}/{TOTAL_SLOTS})")
    elif payload == "FREE":
        current_cars -= 1
        if current_cars < 0: current_cars = 0 # กันเลขติดลบ
        print(f"--> รถออกไปแล้ว! (รถทั้งหมด: {current_cars}/{TOTAL_SLOTS})")
    
    # คำนวณที่ว่าง
    available = TOTAL_SLOTS - current_cars
    print(f"=== เหลือที่ว่าง: {available} คัน ===\n")

# เริ่มการทำงาน
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    # เชื่อมต่อกับ localhost (ตัวเอง)
    client.connect("localhost", 1883, 60)
    client.loop_forever() # รันไปเรื่อยๆ
except KeyboardInterrupt:
    print("Program Stopped")