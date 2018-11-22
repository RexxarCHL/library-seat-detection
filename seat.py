import numpy as np
import cv2
from seat_utils import rectangle_overlap, rectangle_area, SeatStatus
from background_subtractor import BackgroundSubtractor


class Seat:
    def __init__(self, initial_background, bb_coordinates, bg_sub_threshold=50, bg_alpha=0.99):
        '''Initialize the object
        # Arguments:
            initial_background: the initial background to store in the object
            bb_coordinates: bounding box coordinates (x0, y0, x1, y1)
            bg_sub_threshold: binary thresholding value for the background subtractor
            bg_alpha: learning rate for the background subtractor. 1 means no update and 0 means no memory.
        '''
        self.status = SeatStatus.EMPTY  # Status of the seat. One of EMPTY, ON_HOLD, or OCCUPIED
        self.bb_coordinates = tuple(bb_coordinates)  # Bounding box coordinates in the full frame
        self.bb_area = rectangle_area(bb_coordinates)

        self.trackers = [  # Trackers for tracking objects for seats that are ON_HOLD
            cv2.TrackerMOSSE_create(),
            cv2.TrackerMOSSE_create()]
        self.trackers_bb = [None for _ in self.trackers]
        self.trackers_status = [False for _ in self.trackers]
        self.chair_tracker = cv2.TrackerMOSSE_create()
        self.chair_tracker_bb = None
        self.chair_tracker_status = False

        self.person_in_frame_counter = 0  # Counter that increments if a person is detected in the seat
        self.empty_seat_counter = 0  # Counter the increments if the seat is empty, i.e. no person or objects detected
        self.skip_counter = 0  # Track skipped frames for handling bouncing detection boxes
        self.MAX_SKIP_FRAMES = 30  # Maximum frames allowed before counters reset
        self.MAX_EMPTY_FRAMES = 100
        self.TRANSITION_FRAMES_THRESHOLD = 100  # Frames needed for state transition

        self.background = initial_background
        self.empty_chair_bb = None
        self.current_chair_bb = None
        self.bg_subtractor = BackgroundSubtractor(initial_background, 50, 0.99)
        # self.OCCUPIED_PERCENTAGE = 0.3
        self.CONTOUR_AREA_THRESHOLD = 2500

    def get_seat_image(self, frame):
        '''Crop the frame to get the image of the seat'''
        x0, y0, x1, y1 = self.bb_coordinates
        return frame[y0:y1, x0:x1]

    def person_detected(self):
        '''Increment counter when a person is detected in the frame. Transition if conditions are met'''
        self.skip_counter = 0
        if self.status is SeatStatus.EMPTY:
            self.person_in_frame_counter += 1
            if self.person_in_frame_counter == self.TRANSITION_FRAMES_THRESHOLD:
                self.become_occupied()
        elif self.status is SeatStatus.ON_HOLD:
            # TODO
            pass
        else:  # SeatStatus.OCCUPIED
            if self.person_in_frame_counter < self.MAX_EMPTY_FRAMES:
                self.person_in_frame_counter = self.MAX_EMPTY_FRAMES

    def no_person_detected(self, seat_img):
        '''Increment empty seat counter when a person is not present. Transition if conditions are met'''
        # if self.status is SeatStatus.OCCUPIED:
        #     # Probably a bouncing detection
        #     self.skip_counter += 1

        # if self.status is not SeatStatus.EMPTY:
        #     self.empty_seat_counter += 1
        #     if self.empty_seat_counter > self.TRANSITION_FRAMES_THRESHOLD:
        #         self.become_empty()
        if self.status is not SeatStatus.EMPTY:
            leftover_obj_bb = self.check_leftover_obj(seat_img)
            if not leftover_obj_bb:  # No leftover objects
                self.person_in_frame_counter -= 1
                if self.person_in_frame_counter == 0:
                    self.become_empty()
            else:  # Some objects are on the seat
                self.become_on_hold()
        else:  # SeatStatus.EMPTY
            if self.skip_counter < self.MAX_SKIP_FRAMES:  # Debounce
                self.skip_counter += 1
            elif self.person_in_frame_counter > 0:
                self.person_in_frame_counter -= 1
            else:  # person in frame counter is 0
                self.update_background(seat_img)

    def become_occupied(self):
        '''Do necessary operations for the seat to become OCCUPIED'''
        # self.reset_counters()
        self.person_in_frame_counter = self.MAX_EMPTY_FRAMES
        self.skip_counter = 0
        self.status = SeatStatus.OCCUPIED  # State transition

    def become_empty(self):
        '''Do necessary operations for the seat to become EMPTY'''
        self.skip_counter = 0
        self.status = SeatStatus.EMPTY  # State transition

    def become_on_hold(self):
        '''Do necessary operations for the seat to become ON_HOLD'''
        # TODO
        self.status = SeatStatus.ON_HOLD  # State transition

    # def reset_counters(self):
    #     # Reset counters
    #     self.person_in_frame_counter = 0
    #     self.empty_seat_counter = 0
    #     self.skip_counter = 0

    # def check_chair_tracking(self, seat_img, chair_bb):
    #     '''Check if the traker for chair have drifted'''
    #     obj_x0, obj_y0, obj_x1, obj_y1 = chair_bb
    #     track_x0, track_y0, track_x1, track_y1 = self.chair_tracker_bb

    #     # Return false if tracker bounding box is entirely within detected bounding box
    #     if (track_x0 > obj_x0) and (track_y0 > obj_y0) and (track_x1 < obj_x1) and (track_y1 < obj_y1):
    #         return False
    #     return True

    # def track_chair(self, seat_img, bbox):
    #     '''Reinitialize chair tracker with a bounding box in the cropped seat image'''
    #     self.chair_tracker = cv2.TrackerMOSSE_create()
    #     ok = self.chair_tracker.init(seat_img, bbox)
    #     self.chair_tracker_status = ok
    #     self.chair_tracker_bb = bbox
    #     print("track_chair: Reinitialize chair tracker returns {}".format(ok))

    # def update_chair_tracker(self, seat_img):
    #     ''' Update the chair tracker
    #     # Retruns: A tuple in the form of (tracker status, bounding box) for the chair
    #     '''
    #     ok, (x0, y0, h, w) = self.chair_tracker.update(seat_img)
    #     bbox = (int(x0), int(y0), int(x0+h), int(y0+w))  # Convert the bounding box coordinates to [x0, y0, x1, y1]

    #     # Update internal tracker status
    #     self.chair_tracker_bb = bbox
    #     self.chair_tracker_status = ok

    #     return (ok, bbox)

    def update_chair_bb(self, bbox):
        self.current_chair_bb = bbox
        if self.status == SeatStatus.EMPTY:
            self.empty_chair_bb = bbox

    def check_leftover_obj(self, seat_img):
        foreground = self.bg_subtractor.apply(seat_img)
        return self.bg_subtractor.get_bounding_rectangles_from_foreground(foreground, self.CONTOUR_AREA_THRESHOLD)

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
            self.trackers_status[tracker_id] = ok
            self.trackers_bb[tracker_id] = bbox
            print("track_object: Reinitialize tracker {} returns {}".format(tracker_id, ok))

    def update_trackers(self, seat_img):
        '''Update the trackers stored in the object
        # Arguments:
            seat_img: Current seat image
        # Returns: A list of tuples in the form of (tracker status, bounding box), i.e. (ok, [x0, y0, x1, y1])
        '''
        rv = []
        for i, tracker in enumerate(self.trackers):
            ok, (x0, y0, h, w) = tracker.update(seat_img)
            bbox = (int(x0), int(y0), int(x0+h), int(y0+w))  # Convert the bounding box coordinates to [x0, y0, x1, y1]

            # Update internal tracker status
            self.trackers_bb[i] = bbox
            self.trackers_status[i] = ok

            # Append to return value
            rv += [(ok, bbox)]

        return rv

    def update_background(self, seat_img):
        '''Update the stored background'''
        if seat_img.shape != self.background.shape:
            raise ValueError("update_background: Incoming img and stored background must match dimensions.")
        self.bg_subtractor.update_background(seat_img)
