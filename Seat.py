import numpy as np
import cv2
from enum import Enum


class SeatStatus(Enum):
    EMPTY = 0
    ON_HOLD = 1
    OCCUPIED = 2


class Seat:
    def __init__(self, initial_background):
        self.status = SeatStatus.EMPTY  # Status of the seat. One of EMPTY, ON_HOLD, or OCCUPIED
        self.trackers = [  # Trackers for tracking objects for seats that are ON_HOLD
            cv2.TrackerMOSSE_create(),
            cv2.TrackerMOSSE_create()]
        self.person_in_frame_counter = 0  # Counter that increments if a person is detected in the seat
        self.empty_seat_counter = 0  # Counter the increments if the seat is empty, i.e. no person or objects detected
        self.skip_counter = 0  # Track skipped frames for handling bouncing detection boxes
        self.MAX_SKIP_FRAMES = 30  # Maximum frames allowed before counters reset
        self.TRANSITION_FRAMES_THRESHOLD = 1500  # Frames needed for state transition
        self.background = initial_background
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    def person_detected(self):
        '''Increment counter when a person is detected in the frame. Transition if conditions are met'''
        if self.status is not SeatStatus.OCCUPIED:
            self.person_in_frame_counter += 1
            if self.person_in_frame_counter > self.TRANSITION_FRAMES_THRESHOLD:
                self.become_occupied()

    def no_person_detected(self):
        '''Increment empty seat counter when a person is not present. Transition if conditions are met'''
        if self.status is not SeatStatus.EMPTY:
            self.empty_seat_counter += 1
            if self.empty_seat_counter > self.TRANSITION_FRAMES_THRESHOLD:
                self.become_empty()

    def become_occupied(self):
        '''Do necessary operations for the seat to become OCCUPIED'''
        # Reset counters
        self.empty_seat_counter = 0
        self.skip_counter = 0

        # State transition
        self.status = SeatStatus.OCCUPIED

    def become_empty(self):
        '''Do necessary operations for the seat to become EMPTY'''
        # Reset counters
        self.person_in_frame_counter = 0
        self.skip_counter = 0

        # State transition
        self.status = SeatStatus.EMPTY

    def track_object(self, tracker_id, img, bbox):
        '''Reinitialize a tracker to track an object in the bounding box in the image
        # Arguments:
            tracker_id: One of {0, 1}. Index for the self.trackers array.
            img: The image that has the object to track
            bbox: Tuple of coordinates of the bounding box in the form of [x, y, h, w]
        '''
        if tracker_id > len(self.trackers) or tracker_id < 0:
            raise ValueError("track_object: tracker_id is not valid.")
            ok = self.trackers[tracker_id].init(img, bbox)
            print("track_object: Updating tracker {} returns {}".format(tracker_id, ok))

    def update_trackers(self, img):
        '''Update the trackers stored in the object
        # Arguments:
            img: Current frame
        # Returns: A list of tuples in the form of (tracker status, bounding box), i.e. (ok, [x, y, h, w])
        '''
        rv = []
        for tracker in self.trackers:
            ok, bbox = tracker.update(img)
            rv += [(ok, bbox)]

        return rv

    def update_background(self, img):
        '''Update the stored background'''
        if img.shape != self.background.shape:
            raise ValueError("update_background: Incoming img and stored background must match dimensions.")
        self.background = img
