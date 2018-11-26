import numpy as np
import cv2
from seat_utils import rectangle_area, SeatStatus, get_overlap_rectangle
from background_subtractor import BackgroundSubtractorMOG2


class Seat:
    def __init__(self, initial_background, bb_coordinates, table_bb, bg_sub_threshold=50, bg_alpha=0.99):
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

        # Get overlapping rectangle with the table
        # seat_x0, seat_y0, seat_x1, seat_y1 = self.bb_coordinates
        # bb_x0, bb_y0, bb_x1, bb_y1 = table_bb

        # Crop the chair bounding box to be within the seat bounding box
        # x0, y0 = max(seat_x0, bb_x0), max(seat_y0, bb_y0)
        # width = min(seat_x1, bb_x1) - x0
        # height = min(seat_y1, bb_y1) - y0
        # x0, y0 = x0 - seat_x0, y0 - seat_y0
        # self.table_bb = x0, y0, x0+width, y0+height

        # Crop the chair bounding box to be within the seat bounding box
        self.table_bb = get_overlap_rectangle(self.bb_coordinates, table_bb, relative=True)
        if self.table_bb is None:
            raise Exception("Seat.__init__: table bounding box is None! "
                            "Are you sure seat bounding box and table bounding box overlap?")

        self.person_in_frame_counter = 0  # Counter that increments if a person is detected in the seat
        self.empty_seat_counter = 0  # Counter the increments if the seat is empty, i.e. no person or objects detected
        self.object_in_frame_counter = 0
        self.skip_counter = 0  # Track skipped frames for handling bouncing detection boxes
        self.MAX_SKIP_FRAMES = 30  # Maximum frames allowed before counters reset
        self.MAX_EMPTY_FRAMES = 50
        self.MAX_OBJECT_FRAMES = 50
        self.TRANSITION_FRAMES_THRESHOLD = 50  # Frames needed for state transition

        self.seat_img = initial_background
        self.initial_chair_bb = None
        self.empty_chair_bb = None
        self.current_chair_bb = None
        table_background = self.get_table_image(initial_background)
        self.bg_subtractor = BackgroundSubtractorMOG2(table_background)
        # self.OCCUPIED_PERCENTAGE = 0.3
        self.CONTOUR_AREA_THRESHOLD = 1800

        self.empty_chair_bb_buffer = None
        self.empty_chair_bb_buffer_head = 0
        self.current_chair_bb_buffer = None
        self.current_chair_bb_buffer_head = 0
        self.BB_BUFFER_SIZE = 60

    def get_seat_image(self, frame):
        '''Crop the frame to get the image of the seat'''
        x0, y0, x1, y1 = self.bb_coordinates
        self.seat_img = frame[y0:y1, x0:x1]
        return self.seat_img

    def get_table_image(self, seat_img):
        x0, y0, x1, y1 = self.table_bb
        # self.table_img = seat_img[y0:y1, x0:x1]
        # return self.table_img
        return seat_img[y0:y1, x0:x1]

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
            leftover_obj_bb = self.check_leftover_obj(seat_img, self.CONTOUR_AREA_THRESHOLD)
            leftover_obj_bb = leftover_obj_bb.any()
            if not leftover_obj_bb:  # No leftover objects
                self.person_in_frame_counter -= 1
                if self.person_in_frame_counter == 0:
                    self.become_empty()
            else:  # Some objects are on the seat
                self.become_on_hold()
        elif self.status is SeatStatus.ON_HOLD:
            leftover_obj_bb = self.check_leftover_obj(seat_img, self.CONTOUR_AREA_THRESHOLD)
            leftover_obj_bb = leftover_obj_bb.any()
            if self.person_in_frame_counter > 0:
                    self.person_in_frame_counter -= 1
            if not leftover_obj_bb:  # No leftover objects
                if self.object_in_frame_counter == 0:
                    self.become_empty()
                else:
                    self.object_in_frame_counter -= 1
            else:  # Some objects are on the seat
                if self.object_in_frame_counter < self.MAX_OBJECT_FRAMES:
                    self.object_in_frame_counter += 1

        else:  # SeatStatus.EMPTY
            # Detect whether there is a new object on the table
            leftover_obj_bb = self.check_leftover_obj(seat_img, self.CONTOUR_AREA_THRESHOLD*1.5)
            leftover_obj_bb = leftover_obj_bb.any()
            if not leftover_obj_bb:  # No leftover objects
                self.update_background(seat_img)
                if self.object_in_frame_counter > 0:
                    self.object_in_frame_counter = 0
            else:
                self.object_in_frame_counter += 1
                if self.object_in_frame_counter == self.TRANSITION_FRAMES_THRESHOLD:
                    self.become_on_hold()

            if self.skip_counter < self.MAX_SKIP_FRAMES:  # Debounce
                self.skip_counter += 1
            elif self.person_in_frame_counter > 0:
                self.person_in_frame_counter -= 1
            else:  # person in frame counter is 0
                pass  # Do nothing (maybe)

    def become_occupied(self):
        '''Do necessary operations for the seat to become OCCUPIED'''
        # self.reset_counters()
        self.person_in_frame_counter = self.MAX_EMPTY_FRAMES
        self.skip_counter = 0
        self.object_in_frame_counter = 0
        self.status = SeatStatus.OCCUPIED  # State transition

    def become_empty(self):
        '''Do necessary operations for the seat to become EMPTY'''
        self.skip_counter = 0
        self.object_in_frame_counter = 0
        self.status = SeatStatus.EMPTY  # State transition

    def become_on_hold(self):
        '''Do necessary operations for the seat to become ON_HOLD'''
        self.object_in_frame_counter = self.MAX_OBJECT_FRAMES
        self.status = SeatStatus.ON_HOLD  # State transition

    def update_chair_bb(self, bbox):
        # seat_x0, seat_y0, seat_x1, seat_y1 = self.bb_coordinates
        # seat_x0, seat_y0, seat_x1, seat_y1 = self.table_bb  # Update if the chair is within the table bounding box
        # bb_x0, bb_y0, bb_x1, bb_y1 = bbox

        # if seat_x1 < bb_x0 or bb_x1 < seat_x0 or seat_y1 < bb_y0 or bb_y1 < seat_y0:
        #     # No intersection
        #     return  # Do nothing

        # # Crop the chair bounding box to be within the seat bounding box
        # x0, y0 = max(seat_x0, bb_x0), max(seat_y0, bb_y0)
        # width = min(seat_x1, bb_x1) - x0
        # height = min(seat_y1, bb_y1) - y0
        # x0, y0 = x0 - seat_x0, y0 - seat_y0

        # bbox = x0, y0, x0+width, y0+height
        chair_bb = get_overlap_rectangle(self.table_bb, bbox)
        # print(chair_bb, self.table_bb, bbox)
        if chair_bb is None:
            return  # No overlap

        # print(chair_bb)

        # # Update the stored bounding boxes
        self.current_chair_bb = chair_bb
        if self.status == SeatStatus.EMPTY:
            self.empty_chair_bb = chair_bb
            if self.initial_chair_bb is None:
                self.initial_chair_bb = chair_bb

        # Update the bounding box in the ring buffer
        if self.current_chair_bb_buffer is not None:
            self.current_chair_bb_buffer[self.current_chair_bb_buffer_head] = chair_bb
            self.current_chair_bb_buffer_head = (self.current_chair_bb_buffer_head + 1) % self.BB_BUFFER_SIZE
        else:
            self.current_chair_bb_buffer = np.array([chair_bb for _ in range(self.BB_BUFFER_SIZE)])

        if self.status == SeatStatus.EMPTY:
            if self.empty_chair_bb_buffer is not None:
                self.empty_chair_bb_buffer[self.empty_chair_bb_buffer_head] = chair_bb
                self.empty_chair_bb_buffer_head = (self.empty_chair_bb_buffer_head + 1) % self.BB_BUFFER_SIZE
            else:
                self.empty_chair_bb_buffer = np.array([chair_bb for _ in range(self.BB_BUFFER_SIZE)])

    def check_leftover_obj(self, seat_img, threshold):
        table_img = self.ignore_chair_in_background(seat_img)
        # table_img = self.get_table_image(seat_img)
        foreground = self.bg_subtractor.get_foreground(table_img)
        # foreground = self.ignore_chair(foreground)
        # return self.bg_subtractor.get_bounding_rectangles_from_foreground(foreground, threshold)
        return self.bg_subtractor.get_leftover_object_mask(foreground, threshold)

    def update_background(self, seat_img):
        '''Update the stored background'''
        table_img = self.ignore_chair_in_background(seat_img)
        # table_img = self.get_table_image(seat_img)
        self.bg_subtractor.apply(table_img)

    def ignore_chair(self, foreground):
        x0, y0, x1, y1 = self.current_chair_bb
        foreground[y0:y1, x0:x1] = 0
        x0, y0, x1, y1 = self.empty_chair_bb
        foreground[y0:y1, x0:x1] = 0
        return foreground

    def ignore_chair_in_background(self, seat_img):
        current_background = self.bg_subtractor.current_background
        table_img = self.get_table_image(seat_img)
        # if self.current_chair_bb is not None:
        #     x0, y0, x1, y1 = self.current_chair_bb
        #     table_img[y0:y1, x0:x1] = current_background[y0:y1, x0:x1]
        # if self.empty_chair_bb is not None:
        #     x0, y0, x1, y1 = self.empty_chair_bb
        #     table_img[y0:y1, x0:x1] = current_background[y0:y1, x0:x1]
        if self.initial_chair_bb is not None:
            x0, y0, x1, y1 = self.initial_chair_bb
            table_img[y0:y1, x0:x1] = current_background[y0:y1, x0:x1]

        # if self.current_chair_bb is not None:
        #     x0, y0, x1, y1 = np.average(self.current_chair_bb_buffer, axis=0).astype(np.int32)
        #     table_img[y0:y1, x0:x1] = current_background[y0:y1, x0:x1]
        #     # for i in range(self.BB_BUFFER_SIZE):
        #     #     x0, y0, x1, y1 = self.current_chair_bb_buffer[i]
        #     #     table_img[y0:y1, x0:x1] = current_background[y0:y1, x0:x1]

        # if self.empty_chair_bb_buffer is not None:
        #     x0, y0, x1, y1 = np.average(self.empty_chair_bb_buffer, axis=0).astype(np.uint8)
        #     table_img[y0:y1, x0:x1] = current_background[y0:y1, x0:x1]
            # for i in range(self.BB_BUFFER_SIZE):
            #     x0, y0, x1, y1 = self.empty_chair_bb_buffer[i]
            #     table_img[y0:y1, x0:x1] = current_background[y0:y1, x0:x1]

        return table_img

    # def ignore_human_in_background(self, person_bb):
    #     current_background = self.bg_subtractor.current_background
        
