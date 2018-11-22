import os
import argparse
import numpy as np
import cv2
import seat_utils
from object_detector import ObjectDetector
from seat import Seat
from seat_utils import CvColor


def _parse_args():
    """Read CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=os.path.expanduser("~/src/cv_project/video/MVI_0993.MP4"),
                        help="Path to the video to run seat detection.")
    parser.add_argument("--seat-bb-csv", type=str, default="seat_bb.csv",
                        help="The CSV file containing bounding box coordinates.")
    parser.add_argument("--pretrained-model", type=str, default="models/faster_rcnn_inception_v2/frozen_inference_graph.pb",
                        help="The frozen TF model downloaded from Tensorflow detection model zoo: "
                             "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md")

    args = parser.parse_args()

    return args


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
    OBJ_DETECTION_THRESHOLD = 0.7

    # Initialize Seats object
    seats = []
    for seat in range(num_seats):
        x0, y0, x1, y1 = seat_bounding_boxes[seat]
        seats.append(Seat(initial_background=frame[y0:y1, x0:x1], bb_coordinates=seat_bounding_boxes[seat]))
    SEAT_OVERLAP_THRESHOLD = 0.3

    # Start the seat detection
    while True:
        success, frame = cap.read()  # Read the next frame
        if not success:
            break  # No more frames

        boxes, scores, classes, num = obj_detector.processFrame(frame)  # Feed the image frame through the network

        # Detect humans in the frame and get the bounding boxes
        detected_person_bounding_boxes = []
        for i, box in enumerate(boxes):
            # Class 1 is "person"
            if classes[i] == 1 and scores[i] > OBJ_DETECTION_THRESHOLD:
                detected_person_bounding_boxes += [(box[1], box[0], box[3], box[2])]
                # Visualize
                seat_utils.draw_box_and_text(frame, "human: {:.2f}".format(scores[i]), box, CvColor.BLUE)

        seat_img = [None for _ in range(num_seats)]
        for seat_id, this_seat in enumerate(seats):
            this_seat_img = this_seat.get_seat_image(frame)  # Crop the image to seat bounding box

            # Calculate overlap of the seat with each person bounding box
            person_detected = False
            for person_bb in detected_person_bounding_boxes:
                overlap_percentage = this_seat.calculate_overlap_percentage(person_bb)
                if overlap_percentage > SEAT_OVERLAP_THRESHOLD:
                    # if seat_id == 1:
                    #     from utils import rectangle_area, rectangle_overlap
                    #     print("Seat {}, percentage {}%".format(seat_id, overlap_percentage*100))
                    #     print(this_seat.bb_coordinates)
                    #     print(person_bb)
                    #     print(this_seat.bb_area)
                    #     print(rectangle_area(person_bb))
                    #     print(rectangle_overlap(this_seat.bb_coordinates, person_bb))
                    #     print("*"*30)
                    person_detected = True  # Enough overlap, mark as person detected in the seat
                    break  # Person detected in the seat, no need to check other boxes

            # Update the seat status
            if person_detected:
                this_seat.person_detected()
            else:
                this_seat.no_person_detected()

            # Put the seat status in the cropped image
            seat_utils.put_seat_status_text(this_seat, this_seat_img)
            seat_img[seat_id] = this_seat_img

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
