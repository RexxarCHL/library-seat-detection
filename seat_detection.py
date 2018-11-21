import os
import argparse
import numpy as np
import cv2
from ObjectDetector import ObjectDetector
from Seat import Seat


def _parse_args():
    """Read CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=os.path.expanduser("~/src/cv_project/video/MVI_0991.MP4"),
                        help="Path to the video to run seat detection.")
    parser.add_argument("--seat-bb-csv", type=str, default="seat_bb.csv",
                        help="The CSV file containing bounding box coordinates.")
    parser.add_argument("--pretrained-model", type=str, default="models/ssd_mobilenet_v2/frozen_inference_graph.pb",
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

    # Create the object detector from the frozen model
    obj_detector = ObjectDetector(args.pretrained_model)
    obj_detection_threshold = 0.7

    # Initialize Seats object
    seats = [Seat() for seat in range(num_seats)]

    # Start the seat detection
    while True:
        success, img = cap.read()  # Read the next frame
        if not success:
            break  # No more frames

        
    # Clean up
    obj_detector.close()
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    args = _parse_args()
    main(args)
