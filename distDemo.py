import numpy as np
import cv2
from iou_tracker.iou_tracker import Tracker
import time
import datetime
import pickle


camera_id = 1
font = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.8
skip_ms = 66
iou_thresh = 0.2
nms_thresh = 0.2
height = 1080 
width = 1920
resize_ratio = 1
target_labels = ['person']
direction = 0

#target_video="rtsp://admin:admin@172.16.22.195:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"

target_video = "test2.MP4"

line = [[300/resize_ratio,600/resize_ratio],[1720/resize_ratio,610/resize_ratio]]

line_width = 20

def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def line_intersection(line1,line2):
    A=line1[0]
    B=line1[1]
    C=line2[0]
    D=line2[1]
    #print A,B,C,D
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def checkOrientation(line1,line2):
    A=line1[0]
    B=line1[1]
    C=line2[0]
    D=line2[1]
    return ccw(A,D,C)


def render_detections(im, detections):
    for detection in detections:
    	xmin = int(detection["xmin"])
    	ymin = int(detection["ymin"])
    	xmax = int(detection["xmax"])
    	ymax = int(detection["ymax"])
    	label = detection["class"]
        width = xmax - xmin
        height = ymax - ymin
	
        highlight_W = xmin + len(label) * 14
        highlight_H = ymin + height
        cv2.rectangle(im, (xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymin+height+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, label, (xmin, highlight_H+14), font, font_size, (0,0,0),1)       
    return im



def render_tracks(im, tracks):
    for id_, track in tracks.items():
        xmin, ymin, xmax, ymax = track['bbox']
        xmin = int(xmin)
        ymin = int(ymin)
        width = int(xmax - xmin)
        height = int(ymax - ymin)
        label = str(id_)
        highlight_W = xmin + len(label) * 14
        highlight_H = ymin + height
        cv2.rectangle(im, (xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymin+height+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, label, (xmin, highlight_H+14), font, font_size, (0,0,0),1)       
    return im


def render_tracks(im, tracks,buf,cntIn,cntOut):
    prev = buf[0]
    blue = buf[1]
    red  = buf[2]
    last = cntIn-cntOut
    dIcnt = cntIn
    dOcnt = cntOut
    for id_, track in tracks.items():
        xmin, ymin, xmax, ymax = track['bbox']
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        #width = int(xmax - xmin)
        height = int(ymax - ymin)
        
        rectagleCenterPont = (int((xmin + xmax) /2), int((ymin + ymax) /2))
        #print rectagleCenterPont
        cv2.circle(im, rectagleCenterPont, 1, (0, 0, 255), 5)
        
        if(prev.has_key(id_)):
            #line = [[(xmin + xmax) /2, (ymin + ymax) /2],[prev[id_][0]],prev[id_][1]]
            lineToCheck = [[rectagleCenterPont[0],rectagleCenterPont[1]],prev[id_]]
            #if ((rectagleCenterPont[1]**2-prev[id_][1]**2)>40000):
            #    continue
            #print lineToCheck,id_
            if(line_intersection(redLine,lineToCheck)):
                #print lineToCheck
                if blue.has_key(id_):
                    cntOut += 1
                    blue.pop(id_)
                elif (blue.has_key(id_)==False):
                    red[id_] = [rectagleCenterPont[0],rectagleCenterPont[1]]
            if(line_intersection(blueLine,lineToCheck)):
                #print lineToCheck
                if red.has_key(id_): 
                    cntIn += 1
                    red.pop(id_)
                elif (red.has_key(id_)==False): 
                    blue[id_] = [rectagleCenterPont[0],rectagleCenterPont[1]]
                    
        prev[id_] = [rectagleCenterPont[0],rectagleCenterPont[1]]
        
        label = str(id_)
        highlight_W = xmin + len(label) * 14
        highlight_H = ymax
        cv2.rectangle(im, (xmin,ymin),(xmax,ymax),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymax+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, label, (xmin, highlight_H+14), font, font_size, (0,0,0),1)
    if (direction == 1) :
        cv2.putText(frame, "in : {}".format(str(cntIn)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.putText(frame, "out: {}".format(str(cntOut)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    else :
        cv2.putText(frame, "in : {}".format(str(cntOut)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.putText(frame, "out: {}".format(str(cntIn)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
   # dTcnt = cntIn-cntOut-last
    dIcnt = cntIn-dIcnt
    dOcnt = cntOut-dOcnt
    return im,cntIn,cntOut,dIcnt,dOcnt
  


def parse_detections(detections, target_labels):
    parsed_detections = []
    for detection in detections:
	    if detection['class'] in target_labels:
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

cap = cv2.VideoCapture(target_video)
tracker = Tracker(sigma_iou = iou_thresh,t_max = 5)


buf = [{},{},{}]  
cntIn = 0
cntOut = 0
lw = line_width
blueLine = [[line[0][0]-lw,line[0][1]-lw],[line[1][0]-lw,line[1][1]-lw]]
redLine = [[line[0][0]+lw,line[0][1]+lw],[line[1][0]+lw,line[1][1]+lw]]

last_ms = int(round(time.time() * 1000))

dets = pickle.load(open("test2Dets.pkl","rb"))

detL = list(reversed(dets))


print "width" + str(width)
print "height" + str(height)

while(True):
    # Capture frame-by-frame
#    now_ms = int(round(time.time() * 1000))
#    if (now_ms - last_ms)<skip_ms:
#        cap.grab()
#        continue

    #print (now_ms - last_ms)
#    last_ms = now_ms
    ret, frame = cap.read()
    if frame is None:
 	print("Empty frame received!")
	print(cap.isOpened())
	continue

    frame = cv2.resize(frame,(width/resize_ratio,height/resize_ratio))
    detections = detL.pop()
    parsed_detections = parse_detections(detections,target_labels)
    tracks = tracker.trackDistance(parsed_detections)
    
    print(tracks)
    
    #dI/Ocnt is the change of people count by each frame
    frame,cntIn,cntOut,dIcnt,dOcnt= render_tracks(frame, tracks,buf,cntIn,cntOut)

    cv2.line(frame, (blueLine[0][0],blueLine[0][1]), (blueLine[1][0],blueLine[1][1]), (250, 0, 0), 1) #blue line
    cv2.line(frame, (redLine[0][0],redLine[0][1]), (redLine[1][0],redLine[1][1]), (0, 0, 255), 1)#red line
    if (direction == 1) :
        cv2.putText(frame, "in", ((blueLine[0][0]+blueLine[1][0])/2-10, (blueLine[0][1]+blueLine[1][1])/2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "out", ((redLine[0][0]+redLine[1][0])/2-10, (redLine[0][1]+redLine[1][1])/2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else :
        cv2.putText(frame, "out", ((blueLine[0][0]+blueLine[1][0])/2-10, (blueLine[0][1]+blueLine[1][1])/2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "in", ((redLine[0][0]+redLine[1][0])/2-10, (redLine[0][1]+redLine[1][1])/2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    frame = cv2.resize(frame,(width/2,height/2))
    # Display the resulting frame
    cv2.imshow('distDemo',frame)
    cv2.waitKey(100)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
