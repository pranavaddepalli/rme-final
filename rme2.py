import cv2
import numpy as np
import time

# load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# main loop to capture and display the distorted live feed
cap = cv2.VideoCapture(0) # 0 is the default camera device index

nonblurstart = 0
# blurstart = time.time()
# cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:

    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
    
    res = frame

    for (x, y, w, h) in faces:
        # face_gray = gray[y:y+h, x:x+w]

        roi = frame[y:y+h, x:x+w]
        # roi = cv2.medianBlur(roi,21)
2
        row,col,ch = frame.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(frame)
        # Salt mode
        num_salt = np.ceil(amount * frame.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in frame.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* frame.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in frame.shape]
        out[coords] = 0

        res = out
        
        # res[y:y+frame.shape[0], x:x+frame.shape[1]] = frame
    
    
    res = cv2.flip(res, 1)
    cv2.imshow("a", res)
    
    if cv2.waitKey(1) == ord('q'): # press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
