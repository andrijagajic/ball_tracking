""" Ball tracker """

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt


game_started = False
visualize_predictions = True


def mask_detection_regions(img):
    # Certain regions inside template matching result are masked out since the ball can never 
    # finish in these regions (videos of people, timer, etc.)
    img[15:58, 860:1035] = 1
    img[850:1020,810:1090] = 1
    img[860:900,150:320] = 1
    img[920:1015,75:475] = 1
    img[860:900,1565:1760] = 1
    img[920:1015,1440:1830] = 1

if __name__ == "__main__":
    """Here we define the argumnets for the script."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--video", type=str, default = './part1.mp4', help="Path to video file")
    arg_parser.add_argument("--out_csv", default = './preds.csv', type=str, help="Path to output .csv file")  
    arg_parser.add_argument("--visualize_predictions", default = False, type = bool, help='Visualize bounding box around ball during tracking')
    arg_parser.add_argument("--th_detection", default = 0.03, type = float, help='Threshold used for detection decision')
    arg_parser.add_argument("--th_tracking", default = 0.035, type = float, help='Threshold used for tracking decision')
    arg_parser.add_argument("--iters_reinit", default = 15, type = int, help = 'Number of iterations with wrong tracking predictions before reinitializing bounding box using detection')
    args = arg_parser.parse_args()

    video_file = args.video
    visualize_predictions = args.visualize_predictions
    th_detection = args.th_detection
    th_tracking = args.th_tracking
    iters_reinit = args.iters_reinit
    out_csv = args.out_csv
 
    # Load video
    cap = cv2.VideoCapture(video_file)

    # Get number of frames in video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize three templates with different scales used for template matching
    ball_1 = cv2.imread('./ball2.png')
    #ball_1 = cv2.resize(ball_1, (5,6))
    h_ball_1, w_ball_1 = ball_1.shape[0:2]
    #ball_2 = cv2.resize(ball_1, (4,4))
    ball_2 = cv2.resize(ball_1, (12,13))
    h_ball_2, w_ball_2 = ball_2.shape[0:2]
    ball_3 = cv2.imread('./ball3.png')
    #ball_3 = cv2.resize(ball_3, (4,4))
    h_ball_3, w_ball_3 = ball_3.shape[0:2]

    balls = [ball_1, ball_2, ball_3]
    w_balls = [w_ball_1, w_ball_2, w_ball_3]
    h_balls = [h_ball_1, h_ball_2, h_ball_3]


    # Initialize Channel and Spatial Reliability Tracking algorithm
    tracker = cv2.TrackerCSRT_create()
    
    # Variable denoting whether the appropriate detection has been produced or reinitialization is needed
    success = False
    
    # Counter of number of consecutive false predictions, leading to detection reinitialization 
    false_cnt = 0

    # Initializing video output
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('./output.avi', fourcc, 60.0, (640, 360))
    
    # Variable used for storing ball position predictions
    preds = []
    n_frames = 0
    total_duration = 0
    # Visualize k random frames
    for frame_no in range(0, num_frames):
        n_frames += 1
        begin = time.time()
        if frame_no % 100 == 0:
            print(frame_no)
        # if frame_no > 0 and frame_no % 1000 == 0:
        #     preds_df = pd.DataFrame(preds, columns = ['frame_no', 'ball_x', 'ball_y'])
        #     preds_df.to_csv('./out_csv.csv', index=False)

        # Read frame
        ret, frame = cap.read()
        # The game has started once the level of green rises above the empirically found threshold 
        mean_g = np.mean(frame[:,:,1])
        if mean_g > 80:
            game_started = True

        # Frame is blurred in order to obtain more stable pixel values
        frame = cv2.GaussianBlur(frame,(9,9),0)
        
        # frame = cv2.resize(frame, (640, 360))
        
        # Detection algorithm 
        if game_started and not success:
            # For each of the balls, we calculate difference between the template and 
            # regions inside the current frame, and select the most similar part of 
            # frame as a ball, in case the difference is lower than the predefined threshold 
            for k in range(0,len(balls)):
                ball = balls[k]
                res = cv2.matchTemplate(frame,ball,cv2.TM_SQDIFF_NORMED)
                mask_detection_regions(res)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                print(min_val)
                if min_val < th_detection: 
                    w_temp = w_balls[k]
                    h_temp = h_balls[k]
                    temp = ball
                    break
            
            # Initialize tracker and draw rectangle around ball detection
            if min_val < th_detection:
                success = True
                top_left = min_loc
                bottom_right = (top_left[0] + w_temp, top_left[1] + h_temp)
                if top_left[0] < 0 or top_left[1] <0 or bottom_right[0] >= frame.shape[1] or bottom_right[1] >= frame.shape[0]:
                    success = False
                if success:
                    cv2.rectangle(frame,top_left, bottom_right, 255, 2)
                    initBB = [top_left[0], top_left[1], w_temp, h_temp]
                    tracker.init(frame, initBB)
                    pred = [frame_no, top_left[0] + round(w_temp/2), top_left[1] + round(h_temp/2)]
                    preds.append(pred)
                    false_cnt = 0



        # Tracking algorithm
        elif game_started and success:
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                # Draw rectangle around tracking prediction
                (x, y, w, h) = [int(v) for v in box]
                if x < 0 or y <0 or x+w >= frame.shape[1] or y + h >= frame.shape[0]:
                    success = False
                    false_cnt = 0
                 
                if success:
                    ball_pred = frame[y:y+h, x:x+w]
                    ball_pred = ball_pred.copy()
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                        (0, 255, 0), 2)

                    # Check the difference betweeen tracking prediction and template
                    if w >= w_temp and h >= h_temp:
                        res = cv2.matchTemplate(ball_pred,temp,cv2.TM_SQDIFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    elif w <= w_temp and h <= h_temp:
                        res = cv2.matchTemplate(temp, ball_pred, cv2.TM_SQDIFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    else:
                        min_val = 0.1
                    print(min_val)
                    
                    # In case the tracking prediction is much different than template for
                    # a certain number of consecutive iterations, reinitialize algorithm 
                    # by performing detection
                    if min_val > th_tracking:
                        false_cnt += 1
                        if false_cnt >= 15:
                            success = False
                    else:
                        false_cnt = 0
                
                if success:
                    pred = [frame_no, x+round(w/2), y+round(h/2)]
                    preds.append(pred)
            
        # Resizing frame for faster storing
        frame = cv2.resize(frame, (640,360))
        out.write(frame)

        # Visualizing predictions
        if visualize_predictions:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        
        duration = time.time() - begin
        total_duration += duration
        
        if frame_no % 100 == 0:
            print('Avg duration: ' + str(total_duration/n_frames))

    out.release()

    # Saving predictions in a .csv file
    preds = pd.DataFrame(preds, columns = ['frame_no', 'ball_x', 'ball_y'])
    preds.to_csv('./out_csv.csv', index=False)