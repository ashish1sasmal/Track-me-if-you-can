import cv2
import numpy as np
import argparse
import time
import sys
import imutils
from live_cam import live
from imutils.video import FPS

# cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)

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

initBB = None

# vs = cv2.VideoCapture(1)

fps = None

while True:
	frame = live()
	# print(frame.shape[:2])

	# frame = frame[1]
	frame = imutils.resize(frame,width=500)
	(H, W) = frame.shape[:2]
	if initBB is not None:
		(success,box) = tracker.update(frame)
		if success:
			(x,y,w,h) = [int(v) for v in box]
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

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
	key = cv2.waitKey(1) & 0xFF

	if key == ord("s"):
		print("s")
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
		tracker.init(frame, initBB)

		fps = FPS().start()

	elif key == ord("q"):
		break



vs.release()
cv2.destroyAllWindows()
