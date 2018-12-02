import os
import argparse
import numpy as np
import cv2
import seat_utils
from object_detector import ObjectDetector
from seat import Seat, SeatStatus
from seat_utils import CvColor, calculate_overlap_percentage, rectangle_overlap, rectangle_area, get_overlap_rectangle, draw_box_and_text
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
    parser.add_argument("--output", type=str, default="",
                        help="Output file name. Leave blank if no output is needed")
    args = parser.parse_args()

    return args


def main(args):
    # Read in the bounding box coordinates
    if not os.path.isfile(args.seat_bb_csv):
        print("Argument seat-bb-csv is not a file: {}".format(args.seat_bb_csv))
        exit()
    # Each seat bounding box is in the format of [x0, y0, x1, y1]
    seat_bounding_boxes = np.genfromtxt(args.seat_bb_csv, delimiter=',', dtype=np.int)
    # seat_bounding_boxes //= downsample_ratio
    num_seats = len(seat_bounding_boxes) - 1

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
    table_bb = seat_bounding_boxes[0]
    for seat in range(num_seats):
        x0, y0, x1, y1 = seat_bounding_boxes[seat+1]
        seats.append(Seat(frame[y0:y1, x0:x1], seat_bounding_boxes[seat+1], table_bb))
    SEAT_OVERLAP_THRESHOLD = 0.3

    TOTAL_FRAME_COUNT = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    VIDEO_1_FRAME_COUNT = 525.0
    VIDEO_2_FRAME_COUNT = TOTAL_FRAME_COUNT - VIDEO_1_FRAME_COUNT

    progress_bar = tqdm(range(int(TOTAL_FRAME_COUNT)), unit='frames')

    seat_labels = np.full((int(VIDEO_2_FRAME_COUNT), num_seats), -1, dtype=int)

    # JUMP_TO_FRAME = 2100
    # cap.set(cv2.CAP_PROP_POS_FRAMES, JUMP_TO_FRAME)
    # progress_bar.update(JUMP_TO_FRAME)

    if args.output:
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
    # Start the seat detection
    # k = 0  # Skip frame counter
    frame_count = 0
    while True:
        success, frame = cap.read()  # Read the next frame
        if not success:
            break  # No more frames
        progress_bar.update()
        # k += 1
        # if k < 3:  # Run every three frames
        #     continue
        # k = 0
        # frame = cv2.resize(frame, (frame_width, frame_height))
        draw_frame = frame.copy()

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

        # Store the seat status for comparison with ground truth
        this_frame_seat_labels = np.full(num_seats, -1, dtype=int)
        # Store the seat images for visualization
        seat_img = [None for _ in range(num_seats)]
        foreground_img = [None for _ in range(num_seats)]
        for seat_id, this_seat in enumerate(seats):
            this_seat_img = this_seat.get_seat_image(frame)  # Crop the image to seat bounding box
            draw_img = this_seat.get_seat_image(draw_frame)

            # Calculate overlap of the seat with each person bounding box
            person_detected = False
            for person_bb in detected_person_bounding_boxes:
                overlap_percentage = calculate_overlap_percentage(
                                        this_seat.bb_coordinates,
                                        person_bb,
                                        this_seat.bb_area)
                if overlap_percentage > SEAT_OVERLAP_THRESHOLD:
                    person_detected = True  # Enough overlap, mark as person detected in the seat
                    break  # Person detected in the seat, no need to check other boxes

            for chair_bb in detected_chair_bounding_boxes:
                # overlap_area, _ = rectangle_overlap(this_seat.bb_coordinates, chair_bb)
                # if overlap_area > rectangle_area(chair_bb)*0.7:
                #     # This chair is 70% within the seat bounding box
                #     this_seat.update_chair_bb(chair_bb)
                relative_bb = get_overlap_rectangle(this_seat.bb_coordinates, chair_bb, relative=True)
                if relative_bb is not None:
                    this_seat.update_chair_bb(relative_bb)

            # Update the seat status
            if person_detected:
                this_seat.person_detected()
            else:
                this_seat.no_person_detected(this_seat_img)

            # Put the seat status in the cropped image
            x0, y0, x1, y1 = this_seat.table_bb
            foreground = this_seat.check_leftover_obj(this_seat_img, this_seat.CONTOUR_AREA_THRESHOLD)
            # foreground = this_seat.ignore_chair(foreground)
            foreground_img[seat_id] = foreground
            foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)
            cv2.addWeighted(foreground, 0.3, draw_img[y0:y1, x0:x1], 0.7, 0, draw_img[y0:y1, x0:x1])
            seat_utils.put_seat_status_text(this_seat, draw_img)
            seat_img[seat_id] = draw_img

            # Store status for this seat
            this_frame_seat_labels[seat_id] = this_seat.status.value

        # SEE_SEAT = 1
        # cv2.imshow("seat{}".format(SEE_SEAT), foreground_img[SEE_SEAT])

        for seat in range(num_seats):
            x0, y0, x1, y1 = seat_bounding_boxes[seat+1]
            draw_frame[y0:y1, x0:x1] = seat_img[seat]
        
        frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if frame_pos > VIDEO_1_FRAME_COUNT:
            seat_labels[frame_count] = this_frame_seat_labels
            frame_count += 1

        if args.output:
            out.write(draw_frame)
        cv2.imshow("Preview", draw_frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        cv2.imwrite("img/frame_{}.jpg".format(frame_pos),  draw_frame)

    # Video playback ended. Clean up
    progress_bar.close()
    obj_detector.close()
    cap.release()
    if args.output:
        out.release()
    cv2.destroyAllWindows()

    # Store the labels for seats
    np.savetxt("labels.csv", seat_labels, fmt="%s", delimiter=',', header='seat0,seat1,seat2,seat3')


if __name__ == "__main__":
    args = _parse_args()
    main(args)
