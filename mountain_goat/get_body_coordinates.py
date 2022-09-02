import cv2
import os
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# path to image
d_path= "/Users/andrew/Desktop/Cropped_Image.jpg"

def get_pose_image(image_path):
    """
    Input : Path to an image,
    Output : Dictionnary containing  x,y,z coordinates and visibility (ratio of certainty) of
    the body's 4 extremities.
    """

    # ipdb.set_trace()
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_tracking_confidence = 0.5,
        min_detection_confidence=0.5) as pose:
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # ipdb.set_trace()
        if results.pose_landmarks:
        #here 29, 30, 19, 20 corresponds to (x, y,z, visibility) of that
        # part of the pose as par mediapipe.solutions.pose docs

            body_keypoints = {"left_hand_x" : results.pose_landmarks.landmark[19].x,
                            "left_hand_y":results.pose_landmarks.landmark[19].y,
                            "left_hand_z":results.pose_landmarks.landmark[19].z,
                            "left_hand_visibility":results.pose_landmarks.landmark[19].visibility,
                            "right_hand_x" : results.pose_landmarks.landmark[20].x,
                            "right_hand_y":results.pose_landmarks.landmark[20].y,
                            "right_hand_z":results.pose_landmarks.landmark[20].z,
                            "right_hand_visibility":results.pose_landmarks.landmark[20].visibility,
                            "left_foot_X" : results.pose_landmarks.landmark[29].x,
                            "left_foot_y":results.pose_landmarks.landmark[29].y,
                            "left_foot_z":results.pose_landmarks.landmark[29].z,
                            "left_foot_visibility":results.pose_landmarks.landmark[29].visibility,
                            "right_foot_x" : results.pose_landmarks.landmark[30].x,
                            "right_foot_y":results.pose_landmarks.landmark[30].y,
                            "right_foot_z":results.pose_landmarks.landmark[30].z,
                            "right_foot_visibility":results.pose_landmarks.landmark[30].visibility
                            }
            return body_keypoints

        #if missing coordinate, return nan
        body_keypoints_nan= {"left_hand_x" : np.nan,
                "left_hand_y":np.nan,
                "left_hand_z":np.nan,
                "left_hand_visibility":np.nan,
                "right_hand_x" : np.nan,
                "right_hand_y":np.nan,
                "right_hand_z":np.nan,
                "right_hand_visibility":np.nan,
                "left_foot_X" : np.nan,
                "left_foot_y":np.nan,
                "left_foot_z":np.nan,
                "left_foot_visibility":np.nan,
                "right_foot_x" : np.nan,
                "right_foot_y":np.nan,
                "right_foot_z":np.nan,
                "right_foot_visibility":np.nan
                }
        return body_keypoints_nan

print(get_pose_image(d_path))
