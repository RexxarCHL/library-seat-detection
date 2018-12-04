# Seat Status Detection in Library

This is a final project for EN530.461/661 Computer Vision Fall/Winter 2018.

In collaboration with Morgan Hobson(@MorganTHobson), Ruiqing Yin(@RuiYinRay)

## Background
Library seats are a scarce resource. As final exam season starts, a large proportion of students choose to study in the library. However, as students spread out among the maze of levels, locating a vacant seat can become an arduous task. Students also often leave their belongings in a seat to mark it as occupied. This greatly reduces the utilization of available seats. We propose a seat occupancy detection system that leverages existing surveillance camera infrastructure and uses CV techniques in the library to map which seats are available.

We propose two directions in solving this problem: one utilizing pre-trained models and traditional computer vision techniques, and another one that explores end-to-end training in identifying which seats are available.

## Approach 1: Pre-trained model and more traditional CV techniques
Our analysis and decomposition of the task lead us to believe that more traditional CV techniques could solve this task. Following the definitions of seat status, the system only needed to focus on identifying:
- whether there is a person detected in the seat area
- whether there are things left on the table
- whether there are things left on the chair

Since the state of a seat is very closely related to whether a human is detected near the seat, we need a reliable human detection method for the task. However, CV techniques before deep learning for human detection like [Histogram of Oriented Gradients](https://ieeexplore.ieee.org/document/1467360) are prone to false detections, duplicate detections, flickering detections, and are often unable to detect humans in different poses and views. 

Deep convolution neural networks address all these issues and can take advantage of GPU acceleration to make the system run close to real-time speed. We chose models trained on the [COCO Dataset](http://cocodataset.org/) because it is one of the most popular benchmarks for multi-class object detection networks, and the size of the object categories is much smaller and more relevant to our task than other datasets. We used pre-trained models in the [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for our tasks. Specifically, we tried [Mobilenet v2](https://arxiv.org/abs/1801.04381), [Faster R-CNN](https://arxiv.org/abs/1506.01497), and [Mask R-CNN](https://arxiv.org/abs/1703.06870). Although it was very promising to have masks for the detected objects from Mask R-CNN, we found that bounding boxes were sufficient for our task and not worth the additional calculation overhead. Eventually, we settled on using Faster R-CNN as our pre-trained model for human and chair detection for its better detection consistency over Mobilenet and faster computation speed over Mask R-CNN.

The rest of the system was built using building blocks from OpenCV 3.4.3. Seat bounding boxes were defined in a video that includes the chair area and the table area of the seat. Each seat kept its running background using Mixture of Gaussian model. At each frame, we fed the image through the network and filtered the bounding boxes down to only human and chair classes. Each tracked seat had a predesignated region in the camera's view with an associated image for background subtraction. If a human bounding box was detected with significant overlap with a seat's region, then that seat would be marked as occupied. Otherwise, the system would perform background subtraction on the region to check for objects marking the seat as on-hold. Otherwise, the seat would be marked as empty. In order to keep the background image up to date for small changes to the image such as the camera's aperture adjusting, when a seat is marked as empty, its image is used to update the background.

In order to account for small fluctuations in state such as a person walking past a seat or a brief failure of the detection model, we used a build-up/decay system where the formal labeling of the seat's state presented by the system outside of its internal logging only occurs once the new state is held for at least 50 frames.

### Package Dependencies
- Python 3
- OpenCV 3.4.3
- Numpy
- [tqdm](https://tqdm.github.io/)

### Results
- Hard accuracy: 86.37%
- Soft accuracy: 90.25% (considering transition frames)

### Videos

<a href="http://www.youtube.com/watch?feature=player_embedded&v=CuB9HgXosaA
" target="_blank"><img src="http://img.youtube.com/vi/CuB9HgXosaA/0.jpg" 
alt="Detection: Setting 1" width="240" height="180" border="10" /></a>

<a href="http://www.youtube.com/watch?feature=player_embedded&v=6F207wUS-hU
" target="_blank"><img src="http://img.youtube.com/vi/6F207wUS-hU/0.jpg" 
alt="Detection: Setting 2" width="240" height="180" border="10" /></a>

## Approach 2: Using end-to-end training
In addition to our approach utilizing pre-trained CV models, we also set about training our own deep learning model for explicitly labeling seat status. Considering its similarity to our task, we decided to implement [VGG](https://arxiv.org/abs/1409.1556) with additional fully-connected layers for classifying seat states. However, due to the lack of computational power and data, we adjusted the model to a downgraded version of VGG.

Deep convolutional networks take a lot of data to train, especially for end-to-end tasks. In order to train our own model, we manually masked and cropped the seats in our videos and captured seat images from individually sampled frames. Those images are re-sized into 224x224 images for labeling and processing. Additionally, some images are flipped to create more variations in the dataset. However, due to our limited computation power and time, we ended up using only 36000 images collected from one angle.

Due to our limited computation power, we implemented a downgraded version of VGG. The image is passed through a stack of convolution layers, where we use 5x5 and 3x3 filters with a fixed stride is of 2 pixels. The spatial padding of convolution layer input is set to 2 pixels in all directions. 5 dropout layers with $0.2$ dropout probability are added to prevent over-fitting. Max-pooling layers follow the first four dropout layers. The convolution layers are followed by two FC layers. The first one has 4096 channels, the second contains 3 channels (one for each status). All hidden layers are equipped with the ReLu activation and batch normalization to boost training. Cross entropy was used as the loss function as in typical multi-class classification tasks.

### Code
The code for end-to-end training is in [end_to_end_training.ipynb](end_to_end_training.ipynb)

### Results
TODO: Add result from end-to-end training