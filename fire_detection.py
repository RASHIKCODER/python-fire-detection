import cv2
import time
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# =========================================================
# ===================== CONFIG ============================
# =========================================================

MODEL_PATH = "aag.pt"

CONF_THRESHOLD = 0.65          # high confidence only
MIN_FIRE_AREA = 4000           # ignore small light sources
FIRE_CONFIRM_TIME = 1        # seconds fire must persist
CLEAR_DELAY = 2.0              # seconds before clearing fire

# MQTT
BROKER = "m"
PORT = 1883
USERNAME = ""
PASSWORD = ""

# =========================================================
# ===================== MODEL =============================
# =========================================================

model = YOLO(MODEL_PATH)
print("Loaded classes:", model.names)

# =========================================================
# ===================== MQTT ==============================
# =========================================================

client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.connect(BROKER, PORT, 60)
client.loop_start()

# =========================================================
# ===================== CAMERA ============================
# =========================================================

cap = cv2.VideoCapture(0)

# =========================================================
# ===================== ROI LOGIC =========================
# =========================================================

zones = {}
zone_count = 0
drawing = False
ix, iy = -1, -1
current_rect = None

def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, current_rect, zone_count

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_rect = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        zone_count += 1
        zname = f"zone{zone_count}"
        zones[zname] = (ix, iy, x, y)
        print(f"{zname} SET:", zones[zname])
        current_rect = None

cv2.namedWindow("AI FIRE MULTI ZONE")
cv2.setMouseCallback("AI FIRE MULTI ZONE", draw_roi)

# =========================================================
# ===================== STATE =============================
# =========================================================

zone_state = {}        # SAFE / FIRE
fire_start = {}        # fire persistence timer
last_seen = {}         # last fire seen (for clear delay)

print("Camera ON | Draw zones with mouse | Press Q to exit")

# =========================================================
# ===================== MAIN LOOP =========================
# =========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    results = model(frame, verbose=False)

    # Init per-frame occupancy
    zone_detected = {z: False for z in zones}

    # Draw zones
    for z, (x1, y1, x2, y2) in zones.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, z, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if current_rect:
        x1, y1, x2, y2 = current_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # ================= FIRE DETECTION ==================

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id].lower()

            if label != "fire" or conf < CONF_THRESHOLD:
                continue

            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            area = (bx2 - bx1) * (by2 - by1)
            if area < MIN_FIRE_AREA:
                continue

            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            for z, (x1, y1, x2, y2) in zones.items():
                if x1 < cx < x2 and y1 < cy < y2:
                    zone_detected[z] = True
                    last_seen[z] = now

    # ================= STATE MACHINE ==================

    for z in zones:

        if z not in zone_state:
            zone_state[z] = "SAFE"
            fire_start[z] = None
            last_seen[z] = now

        topic = f"factory/{z}/light"

        # ---- FIRE CONFIRMATION ----
        if zone_detected[z]:
            if fire_start[z] is None:
                fire_start[z] = now

            if now - fire_start[z] >= FIRE_CONFIRM_TIME:
                if zone_state[z] != "FIRE":
                    client.publish(topic, "FIRE")
                    zone_state[z] = "FIRE"
                    print("🔥 FIRE CONFIRMED:", z)

        else:
            fire_start[z] = None

            if zone_state[z] == "FIRE" and now - last_seen[z] >= CLEAR_DELAY:
                client.publish(topic, "CLEAR")
                zone_state[z] = "SAFE"
                print("✅ FIRE CLEARED:", z)

    # ================= UI ==================

    y = 30
    for z, state in zone_state.items():
        color = (0, 0, 255) if state == "FIRE" else (0, 255, 0)
        cv2.putText(frame, f"{z}: {state}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        y += 25

    cv2.imshow("AI FIRE MULTI ZONE", frame)

    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
        break

# =========================================================
# ===================== CLEANUP ===========================
# =========================================================

cap.release()
client.loop_stop()
client.disconnect()
cv2.destroyAllWindows()

