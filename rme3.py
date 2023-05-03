import cv2
import numpy as np
import time
import serial

OFFSET = 25 #NUMBER OF FRAMES TO OFFSET
# ser = serial.Serial(port='/dev/cu.usbmodem141401') #SERIAL PORT
# print(ser.name)
# ser.write(b'WE OUT HERE ON THE SERIAL PORTTTTT\n')

time.sleep(2)

DISTORTSPEED = 2; # HOW FAST TO DISTORT (1 IS .1 PER FRAME OUT OF 1)

cap = cv2.VideoCapture(0) # 0 is the default camera device index

buffer = [0 for i in range(OFFSET)] # buffer to hold last 5 seconds
t = 0

facetime = 0 # track how long the face has been on screen

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

brokenmirror = cv2.imread('brokenmirror.png')
brokenmirror = cv2.cvtColor(brokenmirror, cv2.COLOR_BGR2GRAY)

# initalize with five seconds of buffer 
while t < OFFSET:
    ret, frame = cap.read()
    buffer[t] = frame
    t += 1

t = 0

probs = np.random.random(buffer[0].shape[:2])

def distort(image, amt):
    amt = max(0, amt)
    amt = min(1, amt)
    # AMT SHOULD BE BETWEEN 0 AND 1: 0 IS MORE COLOR, 1 IS MORE GRAY
    # output = image.copy()
    # colorspace = image.shape[2]
    # black = np.array([0, 0, 0], dtype='uint8')
    # white = np.array([255, 255, 255], dtype='uint8')
    
    # output[probs < (amt / 2)] = black
    # output[probs > 1 - (amt / 2)] = white

    # # grayscale
    # r, g, b = cv2.split(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    graybgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    output = cv2.addWeighted(image, 1 - amt, graybgr, amt, 0)
    return output


while True:
    grayedout = False
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
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        
        facetime += DISTORTSPEED
        grayedout = True

        # ser.write(b'face')
    
    # otherwise, continue
    else:
        if(grayedout):
            facetime -= DISTORTSPEED
            if(facetime == 0):
                grayedout = False
        else:
            facetime = 0

    
   
    res = distort(res, facetime / 10)

    res = cv2.flip(res, 1)
    
    # res = cv2.addWeighted(res, 0.5, brokenmirror, 0.7, 0)
    
    cv2.imshow("a", res)
    
    if cv2.waitKey(1) == ord('q'): # press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
