from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound
import threading

app = Flask(__name__)

model = load_model("mask_detector_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)

alert_played = False

def play_alert():
    playsound("static/alert.mp3")

def generate_frames():
    global alert_played
    while True:
        success, frame = camera.read()
        if not success:
            break

        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128)) / 255.0
            face = np.reshape(face, (1, 128, 128, 3))

            prediction = model.predict(face)[0][0]

            if prediction > 0.5:
                label = "No Mask"
                color = (0, 0, 255)
                if not alert_played:
                    threading.Thread(target=play_alert).start()
                    alert_played = True
            else:
                label = "Mask"
                color = (0, 255, 0)
                alert_played = False

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
