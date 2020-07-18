import cv2
import numpy as np
import argparse
import time
import sys
import imutils
from live_cam import live
from imutils.video import FPS


def chk(A,B,C):
	s=(C[1]-A[1])*(B[0]-A[0]) - (B[1]-A[1])*(C[0]-A[0])
	if s<=0:
		return True
	else:
		return False

cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS[sys.argv[1]]()
foreground = np.ones((100,100,3),dtype='uint8')*255
initBB = None

vs = cv2.VideoCapture("car2.mp4")

fps = None

while True:
	frame = vs.read()
	# frame = live()
	# print(frame.shape[:2])

	frame = frame[1]
	# print(frame)
	frame = imutils.resize(frame,width=500)
	(H, W) = frame.shape[:2]
	cv2.line(frame,(123,165),(420,222),(255,0,0),2)
	if initBB is not None:
		(success,box) = tracker.update(frame)
		if success:
			alpha=0.4
			(x,y,w,h) = [int(v) for v in box]

			sun = cv2.imread("sun.png")
			res = cv2.resize(sun,(w, h), interpolation = cv2.INTER_CUBIC)
			background = cv2.flip(res,1)
			sun = cv2.addWeighted(background[150:250,150:250,:],alpha,foreground[0:100,0:100,:],1-alpha,0)
			frame[y:y+h, x:x+w] = sun
			frame = cv2.addWeighted(background,0.4,overlay,0.1,0)



		fps.update()
		fps.stop()

		info = [
		("Tracker",sys.argv[1]),
		("Success ","Yes" if success else "No"),
		("FPS",f"{round(fps.fps(),2)}"),
		]

		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF

	if key == ord("s"):
		print("s")
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
		tracker.init(frame, initBB)

		fps = FPS().start()

	elif key == ord("q"):
		break



vs.release()
cv2.destroyAllWindows()

# 154,190
# 420,222
