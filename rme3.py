import cv2
import numpy as np
import time
import serial

OFFSET = 1 #NUMBER OF FRAMES TO OFFSET
# ser = serial.Serial('/dev/cu.usbmodem141401') #SERIAL PORT

cap = cv2.VideoCapture(0) # 0 is the default camera device index

buffer = [0 for i in range(OFFSET)] # buffer to hold last 5 seconds
t = 0

facetime = 0 # track how long the face has been on screen

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# initalize with five seconds of buffer 
while t < OFFSET:
    ret, frame = cap.read()
    buffer[t] = frame
    t += 1

t = 0

def add_noise(image, prob):
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output



while True:
    ret, frame = cap.read()
        
    # write to the buffer
    if t >= OFFSET:
        t = t % OFFSET
    
    res = buffer[t]
    buffer[t] = frame

    t += 1

    detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
    
    # if there's a face, distort
    if (len(faces) > 0):
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ser.write(b'face')
    
    # otherwise, continue
    else:
        ser.write(b'no face')
        res = res 
    
    res = add_noise(res, .3)

    res = cv2.flip(res, 1)
    cv2.imshow("a", res)
    
    if cv2.waitKey(1) == ord('q'): # press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
