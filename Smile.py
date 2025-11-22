import cv2

# -------------------------------------------
# Load Haarcascade Models
# -------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# -------------------------------------------
# Smile Percentage Calculation
# -------------------------------------------
def smile_percent(face_w, face_h, smile_w, smile_h):
    face_area = face_w * face_h
    smile_area = smile_w * smile_h
    percent = int((smile_area / face_area) * 100)
    return percent

# -------------------------------------------
# Start Webcam
# -------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # Detect smile
        smiles = smile_cascade.detectMultiScale(face_gray, 1.7, 20)

        smile_text = "Smile: 0%"

        if len(smiles) > 0:
            # Take the first detected smile
            (sx, sy, sw, sh) = smiles[0]

            cv2.rectangle(face_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

            # Calculate smile percentage
            percent = smile_percent(w, h, sw, sh)
            smile_text = f"Smile: {percent}%"

        # Display text
        cv2.putText(frame, smile_text, (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        break  # Only first face for now

    # Show result
    cv2.imshow("Smile Detection with Percentage", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
