import cv2
import requests
import time
import numpy as np

API_URL = "http://127.0.0.1:8000/upload"  # your FastAPI endpoint

def detect_receipt(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # detect large contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50000:   # threshold to detect a big rectangular paper
            return True
    return False


def auto_capture():
    cam = cv2.VideoCapture(0)

    print("📷 Starting automated receipt scanner...")
    print(" Place the receipt in front of the camera.")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        display = frame.copy()

        # visual indicator in the window
        cv2.putText(display, "Place receipt in the frame to auto-capture...",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Auto Scanner", display)

        if detect_receipt(frame):
            print("📸 Receipt detected! Capturing image...")
            time.sleep(0.8)

            filename = "captured_receipt.jpg"
            cv2.imwrite(filename, frame)

            print("✔ Image captured:", filename)

            # send to backend
            print("Uploading to FastAPI backend...")
            files = {"file": open(filename, "rb")}
            response = requests.post(API_URL, files=files)

            print("📥 Backend response:")
            print(response.json())

            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    auto_capture()
