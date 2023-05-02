import cv2
import numpy as np
import time
import serial

OFFSET = 50 #NUMBER OF FRAMES TO OFFSET
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

probs = np.random.random(buffer[0].shape[:2])

def distort(image, amt):

    output = image.copy()
    colorspace = image.shape[2]
    black = np.array([0, 0, 0], dtype='uint8')
    white = np.array([255, 255, 255], dtype='uint8')
    
    output[probs < (amt / 2)] = black
    output[probs > 1 - (amt / 2)] = white

    # # grayscale
    # r, g, b = cv2.split(image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gr, gg, gb = 100, 100, 100

    # output = np.maximum(0.2989 * r, gr) + np.maximum(0.5870 * g, gg) + np.maximum(0.1140 * b, gb)

    return output

while True:
    ret, frame = cap.read()
        
    # write to the buffer
    if t >= OFFSET:
        t = t % OFFSET
    
    res = buffer[t]
    buffer[t] = frame

    t += 1

    # detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.25, minNeighbors=3, minSize=(30, 30))
    
    # if there's a face, distort
    if (len(faces) > 0):
        # draw a rectangle first
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        
        if facetime < 25:
            facetime += 2
        elif facetime * 1000 >= .1:
            facetime -= 3
        else:
            facetime += 1

        # ser.write(b'face')
    
    # otherwise, continue
    else:
        # ser.write(b'no face')
        res = res 
        facetime -= 3
    
    res = distort(res, facetime / 1000)

    res = cv2.flip(res, 1)
    cv2.imshow("a", res)
    
    if cv2.waitKey(1) == ord('q'): # press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
