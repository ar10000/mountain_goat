import cv2
import os, sys
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



def get_pose_image(image):
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_tracking_confidence = 0.5,
        min_detection_confidence=0.5) as pose:
        # image = cv2.imread(image_path)
        # image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
        #here 29, 30, 19, 20 corresponds to (x, y,z, visibility) of that
        # part of the pose as par mediapipe.solutions.pose docs
            body_keypoints = {"left_hand_x" : results.pose_landmarks.landmark[19].x,
                            "left_hand_y":results.pose_landmarks.landmark[19].y,
                            "right_hand_x" : results.pose_landmarks.landmark[20].x,
                            "right_hand_y":results.pose_landmarks.landmark[20].y,
                            "left_foot_X" : results.pose_landmarks.landmark[29].x,
                            "left_foot_y":results.pose_landmarks.landmark[29].y,
                            "right_foot_x" : results.pose_landmarks.landmark[30].x,
                            "right_foot_y":results.pose_landmarks.landmark[30].y,
                            }
            return body_keypoints
        body_keypoints_nan= {"left_hand_x" : np.nan,
                "left_hand_y":np.nan,
                "right_hand_x" : np.nan,
                "right_hand_y":np.nan,
                "left_foot_X" : np.nan,
                "left_foot_y":np.nan,
                "right_foot_x" : np.nan,
                "right_foot_y":np.nan,
                }


        return body_keypoints_nan
