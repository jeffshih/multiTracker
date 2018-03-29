import cv2
import numpy


class KCFTracker():
    def __init__(self):
        self.id_count = 0
        self.frame = 0
        self.tracks_active = {}
        self.trackers = []
 
    def clearTrack(self):
        targetFrame = self.frame -5
        if self.tracks_active.has_key(targetFrame):
            del(self.tracks_active[targetFrame])

    def createKCFTracker(self,frame,bbox):
        tracker = cv2.cv2.TrackerKCF_create()
        tracker.init(frame,bbox)
        return tracker
    
    def track(dets):
        self.frame +=1
        self.tracks_active[self.frame] = {}
        self.clearTrack()

