# ball_tracking

The solution consists of two parts: detection and tracking. Detection is used to initialize tracker and to help it reinitalize once the tracker loses the target. 
Template matching is used as an algorithm for detection, with three different templates of a ball (with different scales) used. Once the detection is set,
tracking takes place. Tracking is done using Channel and Spatial Reliability Tracking algorithm, built in OpenCV library. The precision of tracking is monitored,
and if the difference between template and tracking prediction gets too high, the algorithm is reinitialized.

The algorithm performs at a ~11FPS frame rate. Even though this is sub-real-time, it can be made to work in real-time by simply sampling every fifth frame in the video, since
60 FPS is more than the algorithm needs to work. The speed can be also further enhanced by downscaling both frame and templates before running an algorithm, but by testing the 
trade-off between prediction accuracy and inference time, I came up to the conclusion that it is not worth rescaling the frame. 

The video showing the output on part1.mp4 video is available at the following link:
