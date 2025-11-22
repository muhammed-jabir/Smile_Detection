# Smile_Detection
A Python OpenCV-based real-time Smile Detection system that detects faces, identifies smile regions, and calculates smile intensity percentage using Haarcascade classifiers. Lightweight, fast, and beginner-friendly.

# 😄 Real-Time Smile Detection with Percentage (Python + OpenCV)

This project is a simple and fast **smile detection system** built using **Python** and **OpenCV**.  
It detects the user's **face**, **smile**, and calculates a **Smile Percentage** based on the size of the smile relative to the face area.

Perfect for beginners, AI demos, mini-projects, and computer vision learners.

---

## 🚀 Features

- 👤 Real-time **face detection**
- 😀 **Smile detection** using Haarcascade
- 📊 **Smile percentage** calculation based on smile size
- 🎥 Works with any webcam
- ⚡ Lightweight — no heavy ML models required

---

## 🛠 Requirements

Install the required packages:

```bash

📁 Code Overview

The smile percentage is calculated as:

smile_percentage = (smile_area / face_area) * 100


Where:

smile_area = width × height of smile box

face_area = width × height of face box

▶️ How to Run

Clone the repository:

git clone https://github.com/yourusername/smile-detection.git


Navigate into the folder:

cd smile-detection


Run the script:

python smile_detect.py


Press Q to exit the window.

📌 Code Used
import cv2

# Load Haarcascade Models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

def smile_percent(face_w, face_h, smile_w, smile_h):
    face_area = face_w * face_h
    smile_area = smile_w * smile_h
    return int((smile_area / face_area) * 100)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(face_gray, 1.7, 20)
        smile_text = "Smile: 0%"

        if len(smiles) > 0:
            (sx, sy, sw, sh) = smiles[0]
            cv2.rectangle(face_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            percent = smile_percent(w, h, sw, sh)
            smile_text = f"Smile: {percent}%"

        cv2.putText(frame, smile_text, (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        break

    cv2.imshow("Smile Detection with Percentage", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

🧠 Future Enhancements

Add eye detection

Add smile intensity graph

Add emoji overlay (🙂, 😄, 🤣)

🤝 Contributing

Pull requests are welcome!

⭐ Support

If you like this project, don't forget to star the repository!


---

If you want, I can also:

✔ Create a **GitHub banner image**  
✔ Create a **project logo**  
✔ Add badges (Python, OpenCV, License)  
✔ Improve README with animated GIF demo  

Just tell me!
pip install opencv-python

