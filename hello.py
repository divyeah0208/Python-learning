import cv2

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load emojis
emoji_happy = cv2.imread("happy_emoji.png", cv2.IMREAD_UNCHANGED)
emoji_sad = cv2.imread("sad_emoji.png", cv2.IMREAD_UNCHANGED)

# Function to detect faces and display emojis
def detect_faces_and_display_emojis(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Display happy emoji
        if w > 0 and h > 0:  # Ensure width and height are positive values
            emoji_happy_resized = cv2.resize(emoji_happy, (w, h))
            for c in range(0, 3):
                frame[y:y+h, x:x+w, c] = emoji_happy_resized[:,:,c] * (emoji_happy_resized[:,:,3]/255.0) +  frame[y:y+h, x:x+w, c] * (1.0 - emoji_happy_resized[:,:,3]/255.0)
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces and display emojis
    frame = detect_faces_and_display_emojis(frame)

    # Display the resulting frame
    cv2.imshow('Emoji Face Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()


