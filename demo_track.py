import cv2
import numpy as np 
import pickle
from iou_tracker.iou_tracker import Tracker

with open("test2Dets.pkl","rb") as f:
    dets = pickle.load(f)
dets = list(reversed(dets))


def parse_detections(detections):
    parsed_detections = []
    for detection in detections:
	score = detection["score"]
	xmin = detection['xmin']
	xmax = detection['xmax']
	ymin = detection['ymin']
	ymax = detection['ymax'] 
	bbox = (xmin, ymin, xmax,  ymax)
                #if ((ymax-ymin)>30) :
                #    parsed_detections.append({"bbox":bbox, "score":score})
	parsed_detections.append({"bbox":bbox, "score":score})	
    return parsed_detections





targetVideo = "test2.MP4"

cap = cv2.VideoCapture(targetVideo)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

bbox = (233,176,62,92)


tracker = cv2.TrackerKCF_create()
flag,frame = cap.read()
p1 = (int(bbox[0]), int(bbox[1]))
p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)


flag = tracker.init(frame,bbox)

while(cap.isOpened()):
    flag,frame = cap.read()
    if flag == False:
        break
    det = dets.pop()
    parsed_det = parse_detections(det)
    frame = cv2.resize(frame,(width/3,height/3))
    if frame is None:
        print ("Empty frame")
        continue
    flag,bbox = tracker.update(frame)
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
