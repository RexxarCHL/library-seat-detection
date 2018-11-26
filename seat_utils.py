import numpy as np
import cv2
from collections import namedtuple
from enum import Enum


class SeatStatus(Enum):
    EMPTY = 0
    ON_HOLD = 1
    OCCUPIED = 2


class CvColor:
    # Color = namedtuple('Color', 'b g r')
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 255, 255)


def draw_box_and_text(img, text, box, color):
    """Draw bounding box and put text on the image
    # Arguments:
        img: The image to draw on
        text: The formatted text to put on the image
        box: The coordinates for the bounding box (x0, y0, x1, y1)
        color: Tuple of (b, g, r) values for the color of the box and text
    """
    cv2.putText(
        img, text,
        (box[1]+10, box[0]+10), cv2.FONT_HERSHEY_PLAIN,
        0.9, color)
    cv2.rectangle(
        img,
        (box[1], box[0]),
        (box[3], box[2]),
        color, 2)


def draw_tracking_object_bounding_box(seat, img):
    """Draw the tracking bounding boxes in the image"""
    if seat.chair_tracker_status:
        draw_box_and_text(img, "Tracking chair", seat.chair_tracker_bb, CvColor.BLACK)
    for i, tracker in enumerate(seat.trackers):
        if seat.trackers_status[i]:
            draw_box_and_text(img, "Tracking object {}".format(i), seat.trackers_bb[i], CvColor.BLACK)


def put_seat_status_text(seat, img):
    """Put seat status text in the image"""
    if seat.status == SeatStatus.EMPTY:
        color = CvColor.GREEN
        status = "EMPTY"
    elif seat.status == SeatStatus.OCCUPIED:
        color = CvColor.RED
        status = "OCCUPIED"
    else:
        color = CvColor.YELLOW
        status = "ON_HOLD"

    cv2.putText(
        img, "Status: {}".format(status),
        (10, 15), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color)
    cv2.putText(
        img, "Person in frame: {}".format(seat.person_in_frame_counter),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color)
    cv2.putText(
        img, "Skip: {}".format(seat.skip_counter),
        (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color)
    cv2.putText(
        img, "Object in frame: {}".format(seat.object_in_frame_counter),
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color)

    h, w, _ = img.shape
    cv2.rectangle(
        img,
        (2, 2),
        (w-2, h-2),
        color, 2)


def draw_seat_seatus_box(seat, frame):
    x0, y0, x1, y1 = seat.bb_coordinates
    if seat.status == SeatStatus.EMPTY:
        color = CvColor.GREEN
    elif seat.status == SeatStatus.OCCUPIED:
        color = CvColor.RED
    else:
        color = CvColor.YELLOW

    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)


def rectangle_area(coordinates):
    """Calculate the rectangle area given by two points (x0, y0, x1, y1)"""
    return (coordinates[2] - coordinates[0]) * (coordinates[3] - coordinates[1])


def rectangle_overlap(a, b):
    """Calculate overlap between two rectangles
    # Arguments:
        a: Two coordinates that defines rectangle 1 in the form of (x0, y0, x1, y1)
        b: Two coordinates that defines rectangle 2 in the form of (x0, y0, x1, y1)
    # Returns: Overlapping area between two rectangles
    """
    # Decompose the coordinates
    a_x0, a_y0, a_x1, a_y1 = a
    b_x0, b_y0, b_x1, b_y1 = b

    if a_x1 < b_x0 or b_x1 < a_x0 or a_y1 < b_y0 or b_y1 < a_y0:
        # No intersection
        return 0, None
    else:
        x0, y0, x1, y1 = get_overlap_rectangle(a, b, relative=True)
        width = x1 - x0
        height = y1 - y0
        return (width * height, (x0, y0, x1, y1))


def calculate_overlap_percentage(rect1, rect2, rect1_area=None, rect2_area=None):
    '''Calculate overlap percentage of the two rectangles'''
    if rect1_area is None:
        rect1_area = rectangle_area(rect1)
    if rect2_area is None:
        rect2_area = rectangle_area(rect2)

    overlap_area, _ = rectangle_overlap(rect1, rect2)
    if overlap_area == 0:
        # No overlap
        return 0.0
    else:
        # Overlap percentage = overlap / (rect1 + rect2 - overlap)
        return overlap_area / (rect1_area + rect2_area - overlap_area)


def get_overlap_rectangle(a, b, relative=False):
    '''Get the overlapping rectangle of the two input rectangles
    # Arguments:
        a: Two coordinates that defines rectangle 1 in the form of (x0, y0, x1, y1)
        b: Two coordinates that defines rectangle 2 in the form of (x0, y0, x1, y1)
        relative: True will return the relative coordinates relative to rectangle a
    # Returns: Two coordinates that defines rectangle 1 in the form of (x0, y0, x1, y1). None if there is no overlap.
    '''
    a_x0, a_y0, a_x1, a_y1 = a
    b_x0, b_y0, b_x1, b_y1 = b

    if a_x1 < b_x0 or b_x1 < a_x0 or a_y1 < b_y0 or b_y1 < a_y0:
        # No intersection
        return None

    x0, y0, x1, y1 = max(a_x0, b_x0), max(a_y0, b_y0), min(a_x1, b_x1), min(a_y1, b_y1)
    if not relative:
        return x0, y0, x1, y1
    else:
        return x0-a_x0, y0-a_y0, x1-a_x0, y1-a_y0
