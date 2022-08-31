import cv2
import os, sys
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


store_images = os.path.join(os.environ.get('PATH'), 'labelled_images')
d_path= "/Users/andrew/code/ar10000/mountain_goat/mountain_goat-body_reco/test_images" # path to directory with images

def pose_rep(d_path):
    IMAGE_FILES = []
    image_path = d_path
    # gets paths of images and stores them in Image_files
    for i in os.listdir(d_path):
        a_path= os.path.join(d_path, i)
        cv2.imread(a_path)
        IMAGE_FILES.append(a_path)

    BG_COLOR = (192, 192, 192) # gray
    #initilaise pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        #min_detection_confidence =
        #min_tracking_confidence =

        min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):# loop through images
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #print(results)

            #if not results.pose_landmarks: # if there are no coordinates found in images , continue to next image
            #continue

            annotated_image = image.copy()## copy of the image
            # draw landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # store them in folder called images labelled

            #keypoints are the coordinates of all the 33 inflexion points / add Z if needed
            keypoints = []

            for index, data_point in enumerate(results.pose_landmarks.landmark):
                keypoints.append({'X': data_point.x,
                                'Y': data_point.y
                                })

            cv2.imwrite(store_images + str(idx) + '.png', annotated_image)
            # Plot pose world landmarks.
            mp_drawing.plot_landmarks(
                results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

            #print(keypoints)
            left_index = keypoints[19]
            right_index = keypoints[20]
            left_foot = keypoints[31]
            right_foot = keypoints[32]

            body_keypoints = {"left_index" : left_index,
                            "right_index" : right_index,
                            "left_foot" : left_foot,
                            "right_foot" : right_foot
                            }
            return body_keypoints
