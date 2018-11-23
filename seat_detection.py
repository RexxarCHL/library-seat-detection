import os
import argparse
import numpy as np
import cv2
import seat_utils
from object_detector import ObjectDetector
from seat import Seat
from seat_utils import CvColor, calculate_overlap_percentage, rectangle_overlap, rectangle_area
from tqdm import tqdm


def _parse_args():
    """Read CLI arguments"""
    parser = argparse.ArgumentParser(description="Library seat status detection using more traditional computer vision methods.")
    parser.add_argument("--video", type=str, default=os.path.expanduser("data/output.mp4"),
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

    # # Detect whether there is humen in the initial background
    # boxes, scores, classes, num = obj_detector.processFrame(frame)  # Feed the image frame through the network
    # detected_person_bounding_boxes = []
    # detected_chair_bounding_boxes = []
    # for i, box in enumerate(boxes):
    #     # Class 1 is "person"
    #     if classes[i] == 1 and scores[i] > OBJ_DETECTION_THRESHOLD:
    #         detected_person_bounding_boxes += [(box[1], box[0], box[3], box[2])]
    # seat_img = [None for _ in range(num_seats)]
    # for seat_id, this_seat in enumerate(seats):
    #     this_seat_img = this_seat.get_seat_image(frame)  # Crop the image to seat bounding box
    #     # Calculate overlap of the seat with each person bounding box
    #     for person_bb in detected_person_bounding_boxes:
    #         overlap_percentage = calculate_overlap_percentage(this_seat.bb_coordinates, person_bb, this_seat.bb_area)
    #         if overlap_percentage > 0.0:
    #             # Human detected in the first frame in the seat bounding box!
    #             this_seat.person_in_background = True
    #             break  # Person detected in the seat, no need to check other boxes

    progress_bar = tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), unit='frames')

    # Start the seat detection
    while True:
        success, frame = cap.read()  # Read the next frame
        draw_frame = frame.copy()
        progress_bar.update()

        if not success:
            break  # No more frames

        boxes, scores, classes, num = obj_detector.processFrame(frame)  # Feed the image frame through the network

        # Detect humen and chairs in the frame and get the bounding boxes
        detected_person_bounding_boxes = []
        detected_chair_bounding_boxes = []
        for i, box in enumerate(boxes):
            # Class 1 is "person"
            if classes[i] == 1 and scores[i] > OBJ_DETECTION_THRESHOLD:
                detected_person_bounding_boxes += [(box[1], box[0], box[3], box[2])]
                # Visualize
                seat_utils.draw_box_and_text(draw_frame, "human: {:.2f}".format(scores[i]), box, CvColor.BLUE)

            elif classes[i] == 62 and scores[i] > OBJ_DETECTION_THRESHOLD:
                detected_chair_bounding_boxes += [(box[1], box[0], box[3], box[2])]
                # Visualize
                seat_utils.draw_box_and_text(draw_frame, "chair: {:.2f}".format(scores[i]), box, CvColor.YELLOW)

        seat_img = [None for _ in range(num_seats)]
        foreground_img = [None for _ in range(num_seats)]
        for seat_id, this_seat in enumerate(seats):
            this_seat_img = this_seat.get_seat_image(frame)  # Crop the image to seat bounding box
            draw_img = this_seat.get_seat_image(draw_frame)

            # Calculate overlap of the seat with each person bounding box
            person_detected = False
            for person_bb in detected_person_bounding_boxes:
                overlap_percentage = calculate_overlap_percentage(this_seat.bb_coordinates, person_bb, this_seat.bb_area)
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

            for chair_bb in detected_chair_bounding_boxes:
                overlap_area = rectangle_overlap(this_seat.bb_coordinates, chair_bb)
                if overlap_area > rectangle_area(chair_bb)*0.7:
                    # This chair is 70% within the seat bounding box
                    this_seat.update_chair_bb(chair_bb)

            # Update the seat status
            if person_detected:
                this_seat.person_detected()
            else:
                this_seat.no_person_detected(this_seat_img)

            # Put the seat status in the cropped image
            # current_chair_bb = this_seat.current_chair_bb
            # current_chair_bb = (current_chair_bb[1], current_chair_bb[0], current_chair_bb[3], current_chair_bb[2])
            # seat_utils.draw_box_and_text(draw_img, "current chair", current_chair_bb, CvColor.BLACK)
            foreground = this_seat.bg_subtractor.get_foreground(this_seat_img)
            foreground = this_seat.ignore_chair(foreground)
            foreground_img[seat_id] = foreground
            foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)
            cv2.addWeighted(foreground, 0.3, draw_img, 0.7, 0, draw_img)
            seat_utils.put_seat_status_text(this_seat, draw_img)
            seat_img[seat_id] = draw_img

        SEE_SEAT = 1
        cv2.imshow("seat{}".format(SEE_SEAT), foreground_img[SEE_SEAT])

        # img = np.copy(frame)
        for seat in range(num_seats):
            x0, y0, x1, y1 = seat_bounding_boxes[seat]
            draw_frame[y0:y1, x0:x1] = seat_img[seat]
        cv2.imshow("Preview", draw_frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Video playback ended. Clean up
    progress_bar.close()
    obj_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
