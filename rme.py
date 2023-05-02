import cv2
import numpy as np

# load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load the smile detection algorithm
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# function to check if a person is smiling
def is_smiling(gray_face):
    smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.7, minNeighbors=30, minSize=(25, 25))
    for (sx, sy, sw, sh) in smiles:
        return True
    return False

# main loop to capture and display the distorted live feed
cap = cv2.VideoCapture(0) # 0 is the default camera device index
while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # split the image into its color channels
    b, g, r = cv2.split(frame)

    # shift the green and blue channels to create chromatic aberration
    rows, cols, _ = frame.shape
    shift_matrix = np.float32([[1, 0, -10], [0, 1, 10]])
    g = cv2.warpAffine(g, shift_matrix, (cols, rows))
    shift_matrix = np.float32([[1, 0, 10], [0, 1, -10]])
    b = cv2.warpAffine(b, shift_matrix, (cols, rows))

    # merge the color channels back together
    color_frame = cv2.merge((b, g, r))

    # check if a face is detected and if the person is smiling
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            if is_smiling(face_gray):
                cv2.imshow('normal live feed', frame)
            else:
                cv2.imshow('distorted live feed', color_frame)
                break
    else:        
        cv2.imshow('distorted live feed', color_frame)
    
    if cv2.waitKey(1) == ord('q'): # press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
