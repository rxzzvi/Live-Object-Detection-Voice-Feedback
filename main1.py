import cv2
import pyttsx3
import time
from ultralytics import YOLO
from threading import Thread

# Load YOLOv8
model = YOLO("yolov8n.pt")

# Initialize Text to Speech
engine = pyttsx3.init()
engine.setProperty("rate", 160)

# Open webcam
cap = cv2.VideoCapture(0)

last_spoken = {}
SPEAK_DELAY = 1  # seconds between repeat announcements


# Speak in background thread so video doesn't freeze
def speak_async(text):
    engine.say(text)
    engine.runAndWait()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    detected_now = []

    current_time = time.time()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if conf > 0.5:
                detected_now.append(label)

                # Draw box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0), 2)

                # Check speak timer
                if label not in last_spoken or current_time - last_spoken[label] > SPEAK_DELAY:
                    last_spoken[label] = current_time
                    Thread(target=speak_async, args=(f"{label} detected",)).start()

    # Remove objects that disappeared
    for obj in list(last_spoken.keys()):
        if obj not in detected_now:
            del last_spoken[obj]

    cv2.imshow("AI Object Detector with Voice", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
