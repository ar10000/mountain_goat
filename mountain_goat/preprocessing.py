
import cv2
import os, sys
import numpy as np
import mediapipe as mp
from get_body_coordinates import get_pose_image
import ipdb

def create_dataframe(directory_path):
    """takes in a directory path folder and returns a  dataframe
    with this kind of format:
    {image_id : V1_id1
    "left_index" : left_index,
    "right_index" : right_index,
    "left_foot" : left_foot,
    "right_foot" : right_foot,
    "frame_id": vid{index}_frame{index}
                            }

    with V{num}_id{num} = frame id in video 1 or 2 ...
    """
    ## getting all the video paths
    videos= []
    # print(videos)
    # ipdb.set_trace()
    for i in os.listdir(directory_path):
        video_path= os.path.join(directory_path, i)
        videos.append(video_path)

    frames_total=[]# this is a list of dictionaries , each dictionary contains (x, y,z, visibility) of a certain body part as in get_pose_image
    # ipdb.set_trace()
    ## loopig through all paths and generating a dataframe
    for vid_index, video_path in enumerate(videos):# index of videos
        for frame_index , frame in enumerate(os.listdir(video_path)):
            frame_path = os.path.join(video_path, frame)
            # print(frame_path) # id of frame in video
            body = get_pose_image(frame_path)
            # print(body)
            body['frame_id']= f'Vid{vid_index}_frame{frame_index}'
            frames_total.append(body)
    # ipdb.set_trace()
    return frames_total

# print(create_dataframe('/home/william/code/ar10000/mountain_goat/videos'))
