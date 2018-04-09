import cv2
from util import iou

class KCFTracker():
    def __init__(self,thresholdL=0.0):
        self.id_count = 0
        self.frame = 0
        self.tracks_active = {}
        self.trackers = {}
        self.thresholdL = thresholdL
        self.KCFTracks = {}



    def iou2(self,bbox1,bbox2):
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]
        (x01,y01,x11,y11) = bbox1
        (x02,y02,x12,y12) = bbox2
        olx0 = max(x01,x02)
        oly0 = max(y01,y02)
        olx1 = min(x11,x12)
        oly1 = min(y11,y12)
        if olx1 - olx0 <=0 or oly1-oly0<=0:
            return 0

        size1 = (x11-x01)*(y11-y01)
        size2 = (x12-x02)*(y12-y02)
        intersection = (olx1-olx0)*(oly1-oly0)
        union = size1+size2-intersection
        return intersection/union

    def clearTrack(self):
        targetFrame = self.frame -2
        if self.tracks_active.has_key(targetFrame):
            del(self.tracks_active[targetFrame])


    def clearTracker(self,id_):
        del(self.trackers[id_])
        print "delete tracker:{}".format(str(id_))

    def createKCFTracker(self,frame,bbox):
        tracker = cv2.cv2.TrackerKCF_create()
        tracker.init(frame,bbox)
        return tracker
    
    
    def retrieveTracks(self):
        tracks = []
        selectedTracks = {}
        for frame in range(self.frame,self.frame-2,-1):
            if frame in self.KCFTracks:
                for id_,track in self.KCFTracks[frame].items():
                    if id_ not in selectedTracks:
                        tracks += self.KCFTracks[frame].items()
                        selectedTracks[id_] = track
        return tracks 


    def track(self,detections,frame):
        self.frame +=1
        self.tracks_active[self.frame] = {}
        self.KCFTracks[self.frame] = {}
        self.clearTrack()
        dets = [det for det in detections if det['score'] >= self.thresholdL]
        #print self.trackers 
        print self.frame
        if (len(self.trackers) > 0):
            for id_ , tracker in self.trackers.items():
                flag,bbox = tracker.update(frame)
                #bbox = (10,10,30,30)
                if (bbox[2]-bbox[0])*(bbox[3]-bbox[1])<500 :
                    self.clearTracker(id_)
                
                self.KCFTracks[self.frame][id_] = bbox
        
             
        for id_,track in self.retrieveTracks():
            #print track
            if len(dets)>0:
                bestMatch = max(dets,key=lambda x:iou(track,x['bbox']))
                #print bestMatch
                if iou(track,bestMatch['bbox']) < 0.3:
                    if self.trackers.has_key(id_):
                        self.clearTracker(id_)
                else:
                    del dets[dets.index(bestMatch)]

        for det in dets:
            self.id_count+=1
            self.trackers[self.id_count] = self.createKCFTracker(frame,det['bbox'])
            self.KCFTracks[self.frame][self.id_count] = det['bbox']
        return self.KCFTracks[self.frame]
