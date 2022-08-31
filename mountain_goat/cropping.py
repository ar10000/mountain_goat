import os
import cv2
from PIL import Image
import numpy as np
import get_body_coordinates


"""
get image
find axis
set center of climber
define extreme screen position
build box around - conditions
create new axis
X is horizontal to the right 0 - 1
y is vertical towards the bottom 0 - 1
"""

store_images = os.path.join(os.environ.get('PATH'), 'labelled_images')
# path to directory with images
d_path= "/Users/andrew/code/ar10000/mountain_goat/mountain_goat-body_reco/test_images"
body_keypoints = get_body_coordinates.pose_rep(d_path)

if body_keypoints['left_foot']['X'] > body_keypoints['right_foot']['X']:
    d_feet_X = body_keypoints['left_foot']['X'] - body_keypoints['right_foot']['X']
    center_X = body_keypoints['right_foot']['X'] + d_feet_X/2
    d_feet_y = body_keypoints['left_foot']['Y'] - body_keypoints['right_foot']['Y']
    center_Y = body_keypoints['right_foot']['Y'] + d_feet_y/2
else :
    d_feet = body_keypoints['right_foot']['X'] - body_keypoints['left_foot']['X']
    center_X = body_keypoints['left_foot']['X'] + d_feet/2
    d_feet_y = body_keypoints['right_foot']['Y'] - body_keypoints['left_foot']['Y']
    center_Y = body_keypoints['left_foot']['Y'] + d_feet_y/2

climber_center = center_X, center_Y
#x_max, x_min, y_max, y_min = 0.9, 0.1, 0.8, 0.2
x_max_crop, x_min_crop, y_max_crop, y_min_crop = center_X + 0.2, center_X - 0.4, center_Y + 0.3, center_Y - 0.3
print(x_max_crop, x_min_crop, y_max_crop, y_min_crop)

#add if statement in relation to borders
