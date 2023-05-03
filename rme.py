1import cv2
import numpy as np
import time


# load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load the smile detection algorithm
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# function to check if a person is smiling
def is_smiling(gray_face):
    return (len(smile_cascade.detectMultiScale(gray_face, scaleFactor=1.7, minNeighbors=30, minSize=(25, 25))) != 0)

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
    
    res = frame

    print(time.time() - nonblurstart)
    if(time.time() - nonblurstart >= 2): # if it's been at least 2 seconds since the smile

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                if is_smiling(face_gray):
                    print("smile")
                    nonblurstart = time.time()
                    res = frame
                else:
                    print("face no smile")
                    res = color_frame
                    break
        else:
            print("no face")
            res = color_frame
    
   
    # cv2.imshow("window", res)
    res = cv2.flip(res, 1)
    cv2.imshow("a", res)
    
    if cv2.waitKey(1) == ord('q'): # press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
