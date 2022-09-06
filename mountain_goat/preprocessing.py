
import cv2
import os, sys
import numpy as np
import pandas as pd
import mediapipe as mp
from mountain_goat.get_body_coordinates import get_pose_image
import ipdb
import re

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
    for i in os.listdir(directory_path):
        video_path= os.path.join(directory_path, i)
        videos.append(video_path)
    #sort  videos so they are in order
    videos.sort(key=lambda f: int(re.sub('\D', '', f)))
    sequences_total=[]# this is a list of dictionaries , each dictionary contains (x, y,z, visibility) of a certain body part as in get_pose_image
    # print(sequences_total)

    ## looping through all paths and generating a dataframe
    for vid_index, video_path in enumerate(videos):# index of videos
        videos_sequence=[]
        #sort the frames so theyre in order
        subfolder_vid_paths= os.listdir(video_path)
        subfolder_vid_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
        framepath_list=[]
        for frame_index , frame in enumerate(subfolder_vid_paths):
            try:
                frame_path = os.path.join(video_path, frame)
                framepath_list.append(frame_path)
                print(f'Creating dataframe for frame {frame_index} out of {len(subfolder_vid_paths)} of video {vid_index} out of {len(videos)}', end='\r')
                body = get_pose_image(frame_path)
                # print(f'body {body}')
                body['frame_id']= f'Vid{vid_index}_frame{frame_index}'
                videos_sequence.append(body)
            except:
                 breakpoint()
        sequences_total.append(pd.DataFrame(videos_sequence))
    return sequences_total
