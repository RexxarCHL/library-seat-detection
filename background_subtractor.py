import argparse
import numpy as np
import cv2
import os


def _parse_args():
    '''Read CLI arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=os.path.expanduser("data/MVI_0993.MP4"),
                        help="Path to the video to run background subtraction.")

    args = parser.parse_args()

    return args


class BackgroundSubtractor:
    def __init__(self, first_frame, threshold, alpha):
        self.background = self.convert_to_grayscale(first_frame)
        self.threshold = threshold
        self.alpha = alpha
        # Get an 5x5 array filled with ones as the kernel for contour finding
        self.kernel = np.ones((5, 5), np.uint8)

    def convert_to_grayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def update_background(self, frame):
        """Update the background using exponential moving average"""
        frame = self.convert_to_grayscale(frame)
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
        self.background = self.alpha * self.background + (1-self.alpha) * frame

    def set_background(self, frame):
        """Update the stored background with the new frame"""
        self.background = self.convert_to_grayscale(frame)

    def apply(self, frame):
        """Subtract the incoming frame with the background to get the foreground"""
        frame = self.convert_to_grayscale(frame)
        frame = cv2.GaussianBlur(frame, (11, 11), 0)

        # self.update_background(frame)

        diff = cv2.absdiff(self.background.astype(np.uint8), frame)
        _, diff = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        return diff

    def find_contour(self, foreground):
        """Find contours on the image"""
        processed_foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, self.kernel)
        # processed_foreground = cv2.dilate(processed_foreground, self.kernel)

        im2, contours, hierarchy = cv2.findContours(processed_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def find_bounding_rectangles(self, contours, area_threshold):
        """Find bounding rectangles whose area is bigger than given threshold"""
        bounding_rect = []
        for c in contours:
            x, y, h, w = cv2.boundingRect(c)
            if h * w < area_threshold:
                continue
            bounding_rect.append((x, y, x+h, y+w))

        return bounding_rect

    def get_bounding_rectangles_from_foreground(self, foreground, area_threshold):
        return self.find_bounding_rectangles(self.find_contour(foreground), area_threshold)


class BackgroundSubtractorMOG2:
    def __init__(self, initial_frame):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(500, 128, False)
        self.bg_subtractor.apply(cv2.GaussianBlur(initial_frame, (11, 11), 0))
        self.kernel = np.ones((5, 5), np.uint8)

    def apply(self, frame):
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
        return self.bg_subtractor.apply(frame)

    def get_foreground(self, frame):
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
        return self.bg_subtractor.apply(frame, learningRate=0)

    def find_contour(self, foreground):
        """Find contours on the image"""
        processed_foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, self.kernel)
        # processed_foreground = cv2.dilate(processed_foreground, self.kernel)

        im2, contours, hierarchy = cv2.findContours(processed_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def find_bounding_rectangles(self, contours, area_threshold):
        """Find bounding rectangles whose area is bigger than given threshold"""
        bounding_rect = []
        for c in contours:
            x, y, h, w = cv2.boundingRect(c)
            if h * w < area_threshold:
                continue
            bounding_rect.append((x, y, x+h, y+w))

        return bounding_rect

    def get_bounding_rectangles_from_foreground(self, foreground, area_threshold):
        return self.find_bounding_rectangles(self.find_contour(foreground), area_threshold)

    def get_leftover_object_mask(self, foreground, area_threshold):
        processed_foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, self.kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_foreground, connectivity=4)

        # return labels

        matching_labels = (stats[:, cv2.CC_STAT_AREA] > area_threshold).nonzero()[0]
        rv = np.zeros(labels.shape, dtype=np.uint8)
        for label in matching_labels:
            if label == 0:
                continue
            rv[labels == label] = 255
        
        return rv


def main(args):
    vid_path = args.video
    if not os.path.isfile(vid_path):
        raise ValueError("background_subtractor: {} is not a file".format(vid_path))

    cap = cv2.VideoCapture(vid_path)  # Open the video file
    if not cap.isOpened():  # Check if the file is opened successfully
        raise ValueError("background_subtractor: VideoCapture failed to open file {}".format(vid_path))

    ret, frame = cap.read()  # Read the first frame
    # bg_subtractor = BackgroundSubtractor(frame, 50, 0.99)  # Create a background subtrator object with the first frame
    bg_subtractor = BackgroundSubtractorMOG2(frame)
    # bg_subtractor = cv2.createBackgroundSubtractorMOG2(100, 16, False)

    while True:
        ret, frame = cap.read()

        # bg_subtractor.update_background(frame)
        foreground = bg_subtractor.apply(frame)

        # contours = bg_subtractor.find_contour(foreground)
        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # bounding_rect = bg_subtractor.find_bounding_rectangles(contours, 2500)
        # bounding_rect = bg_subtractor.get_bounding_rectangles_from_foreground(foreground, 2500)
        # [cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2) for x0, y0, x1, y1 in bounding_rect]
        labels = bg_subtractor.get_leftover_object_mask(foreground, 1000)

        # # Map component labels to hue val
        # label_hue = np.uint8(179*labels/np.max(labels))
        # blank_ch = 255*np.ones_like(label_hue)
        # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # # cvt to BGR for display
        # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # # set bg label to black
        # labeled_img[label_hue==0] = 0

        # cv2.imshow('labeled.png', labeled_img)
        cv2.imshow('labeled.png', labels)

        cv2.imshow('frame', frame)
        # cv2.imshow('foreground', foreground)
        k = cv2.waitKey(30)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
