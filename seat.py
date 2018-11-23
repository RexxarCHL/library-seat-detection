import numpy as np
import cv2
from seat_utils import rectangle_area, SeatStatus
from background_subtractor import BackgroundSubtractorMOG2


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
        self.bg_subtractor = BackgroundSubtractorMOG2(initial_background)
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
            if self.person_in_frame_counter == self.TRANSITION_FRAMES_THRESHOLD:
                self.become_occupied()
            else:
                self.person_in_frame_counter += 1
        else:  # SeatStatus.OCCUPIED
            if self.person_in_frame_counter < self.MAX_EMPTY_FRAMES:
                self.person_in_frame_counter = self.MAX_EMPTY_FRAMES

    def no_person_detected(self, seat_img):
        '''Increment empty seat counter when a person is not present. Transition if conditions are met'''
        if self.status is SeatStatus.OCCUPIED:
            leftover_obj_bb = self.check_leftover_obj(seat_img)
            if not leftover_obj_bb:  # No leftover objects
                self.person_in_frame_counter -= 1
                if self.person_in_frame_counter == 0:
                    self.become_empty()
            else:  # Some objects are on the seat
                self.become_on_hold()
        elif self.status is SeatStatus.ON_HOLD:
            leftover_obj_bb = self.check_leftover_obj(seat_img)
            if not leftover_obj_bb:  # No leftover objects
                self.person_in_frame_counter -= 1
                if self.person_in_frame_counter == 0:
                    self.become_empty()
            else:  # Some objects are on the seat
                pass  # Do Nothing (maybe)

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

    def update_chair_bb(self, bbox):
        seat_x0, seat_y0, seat_x1, seat_y1 = self.bb_coordinates
        bb_x0, bb_y0, bb_x1, bb_y1 = bbox

        # Crop the chair bounding box to be within the seat bounding box
        x0, y0 = max(seat_x0, bb_x0), max(seat_y0, bb_y0)
        width = min(seat_x1, bb_x1) - x0
        height = min(seat_y1, bb_y1) - y0
        x0, y0 = x0 - seat_x0, y0 - seat_y0

        bbox = x0, y0, x0+width, y0+height

        self.current_chair_bb = bbox
        if self.status == SeatStatus.EMPTY:
            self.empty_chair_bb = bbox

    def check_leftover_obj(self, seat_img):
        # TODO: Change to connected component analysis for faster execution
        foreground = self.bg_subtractor.get_foreground(seat_img)
        foreground = self.ignore_chair(foreground)
        return self.bg_subtractor.get_bounding_rectangles_from_foreground(foreground, self.CONTOUR_AREA_THRESHOLD)

    def update_background(self, seat_img):
        '''Update the stored background'''
        self.bg_subtractor.apply(seat_img)

    def ignore_chair(self, foreground):
        x0, y0, x1, y1 = self.current_chair_bb
        foreground[y0:y1, x0:x1] = 0
        x0, y0, x1, y1 = self.empty_chair_bb
        foreground[y0:y1, x0:x1] = 0
        return foreground
