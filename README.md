# ball_tracking

The solution consists of two parts: detection and tracking. Detection is used to initialize tracker and to help it reinitalize once the tracker loses the target. Template matching is used as an algorithm for detection, with three different templates of a ball (in different scales) used. Once the detection is set, tracking takes place. Tracking is done using Channel and Spatial Reliability Tracking algorithm, built in OpenCV library. The precision of tracking is monitored, and if the difference between template and tracking prediction gets too high, the algorithm is reinitialized.

The algorithm performs at a ~11FPS frame rate. Even though this is sub-real-time, it can be made to work in real-time by simply sampling every fifth frame in the video, since
60 FPS is more than the algorithm needs to work. The speed can be also further enhanced by downscaling both frame and templates before running an algorithm, but by testing the 
trade-off between prediction accuracy and inference time, I came up to the conclusion that it is not worth rescaling the frame. 

The program is run by simply running the build_tracker.py script, with no previous installation needed. OpenCV version used in project is 4.5.3. In order to test the script on
different input, argument video should be set to the path pointing to the video. 

The predictions on part1.mp4 video are presented in predictions.csv file. The x and y coordinates of ball are set to the ones representing the center of the ball prediction during the inference.

The video showing the output on part1.mp4 video is available at the following link: https://drive.google.com/file/d/1rMqLBdXeIXNxMb_a4TXaMs_InQXCl4Jb/view?usp=sharing

A part of video output:

https://user-images.githubusercontent.com/41752671/129495149-c7f141d3-c167-4d75-b34d-7f0b15704fa0.mp4


