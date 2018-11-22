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
    elif seat.status == SeatStatus.OCCUPIED:
        color = CvColor.RED
    else:
        color = CvColor.YELLOW

    cv2.putText(
        img, "Status: {}".format(seat.status),
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
        return 0
    else:
        width = min(a_x1, b_x1) - max(a_x0, b_x0)
        height = min(a_y1, b_y1) - max(a_y0, b_y0)
        return width * height
