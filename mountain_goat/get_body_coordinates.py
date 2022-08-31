import cv2
import os, sys
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


store_images = os.path.join(os.environ.get('PATH'), 'labelled_images')
d_path= "/home/william/code/willsketch/test_mountain_goat/images" # path to directory with images
IMAGE_FILES = []
# gets paths of images and stores them in Image_files
for i in os.listdir(d_path):
					a_path= os.path.join(d_path, i)
					IMAGE_FILES.append(a_path)


BG_COLOR = (192, 192, 192) # gray
#initilaise pose
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):# loop through images
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks: # if there are no coordinates found in images , continue to next image
      continue


    annotated_image = image.copy()## copy of the image
    # draw landmarks
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # store them in folder called images labelled

    cv2.imwrite(store_images + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
