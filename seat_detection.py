import os
import argparse
import numpy as np
import cv2
from ObjectDetector import ObjectDetector
from Seat import Seat
from collections import namedtuple

CvColor = namedtuple('CvColor', 'b g r')
BLUE = CvColor(255, 0, 0)
GREEN = CvColor(0, 255, 0)
RED = CvColor(0, 0, 255)
WHITE = CvColor(255, 255, 255)
BLACK = CvColor(0, 0, 0)

def _parse_args():
    """Read CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=os.path.expanduser("~/src/cv_project/video/MVI_0993.MP4"),
                        help="Path to the video to run seat detection.")
    parser.add_argument("--seat-bb-csv", type=str, default="seat_bb.csv",
                        help="The CSV file containing bounding box coordinates.")
    parser.add_argument("--pretrained-model", type=str, default="models/ssd_mobilenet_v2/frozen_inference_graph.pb",
                        help="The frozen TF model downloaded from Tensorflow detection model zoo: "
                             "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md")

    args = parser.parse_args()

    return args


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
    '''Draw the tracking bounding boxes in the image'''
    if seat.chair_tracker_status:
        draw_box_and_text(img, "Tracking chair", seat.chair_tracker_bb, BLACK)
    for i, tracker in enumerate(seat.trackers):
        if seat.trackers_status[i]:
            draw_box_and_text(img, "Tracking object {}".format(i), seat.trackers_bb[i], BLACK)


def main(args):
    # Read in the bounding box coordinates
    if not os.path.isfile(args.seat_bb_csv):
        print("Argument seat-bb-csv is not a file: {}".format(args.seat_bb_csv))
        exit()
    # Each seat bounding box is in the format of [x0, y0, x1, y1]
    seat_bounding_boxes = np.genfromtxt(args.seat_bb_csv, delimiter=',', dtype=np.int)
    num_seats = len(seat_bounding_boxes)

    # Open the video
    if not os.path.isfile(args.video):
        print("Argument video is not a file: {}".format(args.video))
        exit()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError("Failed to open video: {}".format(args.video))
    success, frame = cap.read()  # Read the first frame
    if not success:
        print("Failed to read the first frame from : {}".format(args.video))

    # Create the object detector from the frozen model
    obj_detector = ObjectDetector(args.pretrained_model)
    obj_detection_threshold = 0.7

    # Initialize Seats object
    seats = []
    for seat in range(num_seats):
        x0, y0, x1, y1 = seat_bounding_boxes[seat]
        seats.append(Seat(initial_background=frame[y0:y1, x0:x1], bb_coordinates=seat_bounding_boxes[seat]))

    # Start the seat detection
    while True:
        success, frame = cap.read()  # Read the next frame
        if not success:
            break  # No more frames

        # Detect objects(human/chair) in each seat bounding box
        seat_img = [None for _ in range(num_seats)]
        person_detected = False
        for seat_id, this_seat in enumerate(seats):
            this_seat_img = this_seat.get_seat_image(frame)  # Crop the image to seat bounding box
            boxes, scores, classes, num = obj_detector.processFrame(this_seat_img)  # Feed through the network

            for i, box in enumerate(boxes):
                # Class 1 is "person"
                if classes[i] == 1 and scores[i] > obj_detection_threshold:
                    person_detected = True
                    this_seat.person_detected()
                    # Visualize
                    draw_box_and_text(this_seat_img, "human: {:.2f}".format(scores[i]), box, BLUE)

                # Class 62 is "chair"
                elif classes[i] == 62 and scores[i] > obj_detection_threshold:
                    # Visualize
                    draw_box_and_text(this_seat_img, "chair: {:.2f}".format(scores[i]), box, GREEN)

                    # TODO: Implement chair tracking
                    # if this_seat.chair_tracker_status is False:  # Not tracking a chair
                    #     this_seat.track_chair(this_seat_img, box)
                    # else:
                    #     this_seat.update_chair_tracker(this_seat_img)
                    #     ok = this_seat.check_chair_tracking(this_seat_img, box)
                    #     if not ok:  # Tracker bounding box is entirely within the detected bounding box
                    #         this_seat.track_chair(this_seat_img, box)  # Reinitialize chair tracker

            if person_detected:
                this_seat.person_detected()
            else:
                this_seat.no_person_detected()

            # Draw tracking objects
            draw_tracking_object_bounding_box(this_seat, this_seat_img)
            cv2.putText(
                this_seat_img, "Status: {}".format(this_seat.status), 
                (10, 10), cv2.FONT_HERSHEY_PLAIN,
                1, WHITE)
            cv2.putText(
                this_seat_img, "Person in frame: {}".format(this_seat.person_in_frame_counter),
                (10, 20), cv2.FONT_HERSHEY_PLAIN,
                1, WHITE)
            cv2.putText(
                this_seat_img, "Empty seat: {}".format(this_seat.empty_seat_counter),
                (10, 30), cv2.FONT_HERSHEY_PLAIN,
                1, WHITE)
            seat_img[seat_id] = this_seat_img
            seats[seat_id] = this_seat  # NOTE: Not sure if this is needed

            # if seat_id == 1:
            #     cv2.imshow("seat{}".format(seat_id), this_seat_img)
            #     key = cv2.waitKey(1)
            #     if key & 0xFF == ord('q'):
            #         break
        # SEE_SEAT = 1
        # cv2.imshow("seat{}".format(SEE_SEAT), seat_img[SEE_SEAT])

        img = np.copy(frame)
        for seat in range(num_seats):
            x0, y0, x1, y1 = seat_bounding_boxes[seat]
            img[y0:y1, x0:x1] = seat_img[seat]
        cv2.imshow("Preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Video playback ended. Clean up
    obj_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
