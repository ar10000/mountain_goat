import cv2
import numpy as np
import emoji
from scipy import signal
import matplotlib.pyplot as plt
from mountain_goat.get_body_coordinates import get_pose_image


def resize_image(image,scale):
    """resize image with a given  scale"""
    scale_percent = scale# percent of original size
    img= image
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_image

def get_significant_frames(file_path, resizing_scale=40, show_frames= True):
    """Takes as input a video of a climber and returns the significant frames in
    that  video.
    Significant frames means a frame where a the climber is holding a grip
    output is a list of the significant frames --> list of ndim.arrays
    parameters
    file path: path to video file
    resizing scale: the scale with which to resize the image , default is 40%
    show_frames : plots of angles and smoothed angles showing significant frames,
    default is True, during production this should be set to False
    """
    #capturing frames
    filename = file_path
    frame_list =[]
    vidCap = cv2.VideoCapture(filename)
    video_length = int(vidCap.get(cv2.CAP_PROP_FRAME_COUNT))
    while vidCap.isOpened():
        success, frame = vidCap.read()
        if not success:
            break
        frame_list.append(frame)
    print(f'######## number of frames captured is {len(frame_list)} and video length is {video_length}')
    print(f'######## shape of frames is {frame_list[0].shape}')

    # resizing frames
    resized_frame_list =[]
    scale = resizing_scale
    print('##### resizing frames .....')
    for image in frame_list:
        resized_image = resize_image(image, scale)
        resized_frame_list.append(resized_image)
    print(f'#### resized frames shape is {resized_frame_list[0].shape}')

    # getting significant frames
    length_video = len(resized_frame_list)
    angles_dict={}
    print(f'####### Calculating angle.....')
    for id, frame in enumerate(resized_frame_list):
        if id == length_video-1:
            # break if there are no more frames in the list
            break
        new_frame = frame
        next_frame = resized_frame_list[id +1]

        # get_pose_image returns dict of coordinates , so here we change teh values(the coordinates)into an array for the first
        # and next frame
        new_frame_coordinates=np.array(list(get_pose_image(new_frame).values()))
        next_frame_coordinates = np.array(list(get_pose_image(next_frame).values()))

        #calculating angle between coordinates in frames
        unit_vector_1 = new_frame_coordinates/ np. linalg. norm(new_frame_coordinates)
        unit_vector_2 = next_frame_coordinates / np. linalg. norm(next_frame_coordinates)
        dot_product = np. dot(unit_vector_1, unit_vector_2)
        angle = np. arccos(dot_product)
        angles_dict[f'frame{id} vs frame{id +1}'] = angle
        if id == int(len(resized_frame_list)/2):
            print(emoji.emojize('######### halfway there still calculating :winking_face: ....'))

    #spikes when a climber makes a move , so we need to get the minima
    angles = np.array(list(angles_dict.values()))
    # apply three layers of smoothing
    print(f'##### applying smoothing layers ..')
    smooth_angles = signal.savgol_filter(x=angles, window_length=18, polyorder=3, mode='nearest')
    smoother_angles = signal.savgol_filter(x=smooth_angles, window_length=18, polyorder=3, mode='nearest')
    smoother_angles = signal.savgol_filter(x=smoother_angles, window_length=18, polyorder=3, mode='nearest')
    minima_indices= signal.argrelmin(smoother_angles)

    # plotting angles from frames
    if show_frames:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        ax[0].plot(angles, label ='angles')
        ax[0].set_title('Angles between frames')
        ax[1].plot(smoother_angles, color='r', label='smooth_angles')
        ax[1].set_title('Smooth angles with minimas ')
        ax[1].vlines(minima_indices[0], ymin=0, ymax=0.10, colors='g', label='signf_frames')
        plt.legend()
        plt.show()

    # using indices of maxima's to get significant frames
    significant_frames =[]
    print(f'####### getting significant frames from minima indices...')
    for index in minima_indices[0]:
        significant_frames.append(resized_frame_list[index])

    print(f'number of significant frames found is {len(significant_frames)}')


    return significant_frames

if __name__ == '__main__':
    filepath = 'notebooks/clip2.mp4'
    get_significant_frames(filepath)
