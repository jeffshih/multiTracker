# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------

from time import time
from util import load_mot, iou,calcDist

class Tracker():
    def __init__(self, sigma_l=0, sigma_h=0.5, sigma_iou=0.2, t_max=5, logging = False):
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_max = t_max
        self.frame = 0
        self.id_count = 0
        self.tracks_active = {}

	self.logging = logging
	self.log = {0:[]}
    #Clear the old tracks
    def clean_old_tracks(self):
        target_frame = self.frame - self.t_max
        if self.tracks_active.has_key(target_frame):
            if self.logging:
		self.log[self.frame]=["[Log]: Tracks Deleted:{}".format(self.tracks_active[target_frame].keys())]
	    del(self.tracks_active[target_frame])
	
    #Retrieve tracks in an correct matching order
    def retrieve_tracks(self):
        tracks = []
        selected_tracks = {}
        frames = range(self.frame, self.frame - self.t_max, -1)
        for frame in frames:
            if frame in self.tracks_active:
                for id_, track in self.tracks_active[frame].items():
                    if id_ not in selected_tracks:
                        #tracks += (id_,track)
                        tracks += self.tracks_active[frame].items()
			selected_tracks[id_]=track
                #tracks += self.tracks_active[frame].items()
        return tracks

    def track(self, detections):
        self.frame += 1
        self.tracks_active[self.frame] = {}
        #Clear the tracks in old frame
        self.clean_old_tracks()

        dets = [det for det in detections if det['score'] >= self.sigma_l]
        
        for id_, track in self.retrieve_tracks():
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(track['bbox'], x['bbox']))
                if iou(track['bbox'], best_match['bbox']) >= self.sigma_iou:
                    self.tracks_active[self.frame][id_] = best_match                 
                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

        #Create new tracks
        for det in dets:
            self.id_count += 1
            self.tracks_active[self.frame][self.id_count] = det

        #Return the current tracks 
        return self.tracks_active[self.frame]  

        

    def trackDistance(self,detections):
        self.frame +=1
        self.tracks_active[self.frame] = {}
        self.clean_old_tracks()

        dets = [det for det in detections if det['score'] >= self.sigma_l]

        #mapping tracks
        for id_,track in self.retrieve_tracks():
            if len(dets) > 0:
                best_match = max(dets, key=lambda x: iou(track['bbox'], x['bbox']))
                if iou(track['bbox'], best_match['bbox']) >= self.sigma_iou:
                    self.tracks_active[self.frame][id_] = best_match                 
                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]
        #mapping tracks with distance
        for id_, track in self.retrieve_tracks():
            if len(dets) > 0:
                matches = [det for det in dets if iou(track['bbox'],det['bbox'])>0]
                if len(matches) >0:
                    nearestMatch = min(matches,key=lambda x: calcDist(track['bbox'],x['bbox']))
                    self.tracks_active[self.frame][id_] = nearestMatch
                    del dets[dets.index(nearestMatch)]
        for det in dets:
            self.id_count +=1
            self.tracks_active[self.frame][self.id_count] = det

        return self.tracks_active[self.frame]


def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.

    Args:
         detections (list): list of detections per frame, usually generated by util.load_mot
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        list: list of tracks.
    """

    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
                if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'], best_match['score'])

                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished


def track_iou_matlab_wrapper(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Matlab wrapper of the iou tracker for the detrac evaluation toolkit.

    Args:
         detections (numpy.array): numpy array of detections, usually supplied by run_tracker.m
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        float: speed in frames per second.
        list: list of tracks.
    """

    detections = detections.reshape((7, -1)).transpose()
    dets = load_mot(detections)
    start = time()
    tracks = track_iou(dets, sigma_l, sigma_h, sigma_iou, t_min)
    end = time()

    id_ = 1
    out = []
    for track in tracks:
        for i, bbox in enumerate(track['bboxes']):
            out += [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]),
                    float(track['start_frame'] + i), float(id_)]
        id_ += 1

    num_frames = len(dets)
    speed = num_frames / (end - start)

    return speed, out
