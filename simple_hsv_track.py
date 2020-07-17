
import cv2
import numpy as np
from live_cam import live
import imutils


cv2.namedWindow("Track",cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture("pool.mp4")
ret = False
lower=0
upper=0
pixel = (20,60,80)


def pick_color(event,x,y,flags,param):
    global ret,lower,upper,pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = frame[y,x]
        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(pixel, lower, upper)
        ret = True

cv2.setMouseCallback('Track', pick_color)

def object(hsv):

        global lower,upper
        # lower_y = np.array([ 17,180, 180])
        # upper_y = np.array([ 38, 232, 210])

        lower = np.array(lower)
        upper = np.array(upper)

        mask = cv2.inRange(hsv, lower, upper)

        nask = cv2.erode(mask,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts)>0:
            m = max(cnts,key=cv2.contourArea)
            (x,y), radius = cv2.minEnclosingCircle(m)

            cv2.circle(frame,(int(x),int(y)),int(radius),(0,0,255),2)

        return frame

while True:
    # frame = live()
    rete, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    if ret:
        frame = object(hsv)
    cv2.imshow("Track",frame)

    key = cv2.waitKey(50) & 0xFF
    if key == ord("q"):
        break

# cap.release()
cv2.destroyAllWindows()
