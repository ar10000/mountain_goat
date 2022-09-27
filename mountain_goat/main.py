from email.mime import image
from subprocess import list2cmdline
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from mountain_goat import grip_detection
from mountain_goat import next_move_model
from mountain_goat.get_body_coordinates import get_pose_image
from mountain_goat.grip_detection import get_grips
from mountain_goat.preprocessing import create_dataframe
from mountain_goat.next_move_model import initialize_model, compile_model, train_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ipdb
from mountain_goat.data import get_data
from mountain_goat.params import RAW_DATA_DIR
from tensorflow import keras
from mountain_goat.color_identification import grip_colors

# list_videos = create_dataframe('../raw_data/mountain_goat_screenshots')

def check_nan_videos(videos:list) -> pd.DataFrame:
    """takes in list of dataframes and returns a summary of the videos w.r.t the number of nan values"""
    vid_summary =[]
    for df in videos:
        num_nans= df['right_hand_x'].isnull().sum(axis=0)
        vid_id = df['frame_id'].to_list()[0]
        percent= num_nans/len(df)
        res= {'vid_id': vid_id, 'num_nans': num_nans, 'percent':percent}
        vid_summary.append(res)
    return pd.DataFrame(vid_summary)

def remove_nan(list_videos:list) -> list:
    """takes in alist of videos and removes nan"""
    list_videos_no_nan=[]
    for df in list_videos:
        df.dropna(inplace=True)
        list_videos_no_nan.append(df)
    return list_videos_no_nan

def preprocess(list_videos) -> np.array:
    """preprocess data and returns an X and y
        input is a list of dataframes
        output is X_train: np.array
                  X_test: np.array
                  y_train: np.array
                  y_test: np.array
    """
    y = []
    X = []
    #remove the Nan
    list_videos_no_nan = remove_nan(list_videos)
    for df in list_videos_no_nan:
        df = df.drop(columns='frame_id', axis=1)
        last_frame= df.iloc[-1:]
        frames = df[:-1]
        y.append(np.array(last_frame))
        X.append(np.array(frames))

    # plot distribution of frames per video
    frames_to_plot=[]
    for vid in X:
        frames_to_plot.append(vid.shape[0])
    plt.hist(frames_to_plot)

    # pad sequences to fill up sequences
    X_pad = pad_sequences(X, dtype='float32', padding='post', value=-1000)

    train_split = 0.7
    test_split = 0.3
    number_of_videos = X_pad.shape[0]
    number_train_videos= int(round(number_of_videos*train_split))
    number_test_videos = int(round(number_of_videos*test_split))
    new_y= np.vstack(y)
    X_pad_train = X_pad[:number_train_videos, :,:]
    X_pad_test = X_pad[number_train_videos:,:,:]
    assert type(X_pad_train) is np.ndarray
    assert type(X_pad_test) is  np.ndarray
    y_train=np.array(new_y)[:number_train_videos, :]
    y_test = np.array(new_y)[number_train_videos:, :]

    return X_pad_train, X_pad_test, y_train, y_test

def train():
    """train model """

    data_location = input("Enter data source (cloud or local): ")
    dataset = input("Enter dataset to use (screenshots or UCSD): ")
    get_data(data_location, dataset)

    local_storage_filename_screenshots = os.path.join(RAW_DATA_DIR, "mountain_goat_screenshots")
    local_storage_filename_ucsd = os.path.join(RAW_DATA_DIR, "mountain_goat_UCSD")

    if dataset=='screenshots':
        list_videos = create_dataframe(local_storage_filename_screenshots)
    elif dataset=='UCSD':
        list_videos = create_dataframe(local_storage_filename_ucsd)

    # list_videos = create_dataframe('raw_data/mountain_goat_screenshots')
    #list_videos = create_dataframe(d_path)
    X_pad_train, X_pad_test, y_train, y_test = preprocess(list_videos)
    # ipdb.set_trace()
    model = initialize_model()
    model = compile_model(model)
    model, history = train_model(model, X_pad_train, y_train)
    return model, history

def pred_next_move(X:np.ndarray)-> np.ndarray:
    """make prediction  from series of pictires
    Should take as input array of body coordinates but also array of grip coordinates"""
    # load model from cloud
    model=None
    # check if something is in cloud
    if model is None:
        model , history = train()

    #TODO if we have time make cosine similarity function
    prediction = model.predict(X)
    return prediction





def next_position(grip_model , next_move_model, list_frames=0):
    # img1= cv2.imread('/home/william/code/ar10000/mountain_goat/raw_data/mountain_goat_screenshots/video106/Screenshot 2022-08-29 at 12.13.42.png')
    # img2= cv2.imread('/home/william/code/ar10000/mountain_goat/raw_data/mountain_goat_screenshots/video106/Screenshot 2022-08-29 at 12.13.49.png')
    # img3= cv2.imread('/home/william/code/ar10000/mountain_goat/raw_data/mountain_goat_screenshots/video106/Screenshot 2022-08-29 at 12.14.19.png')
    # img4 = cv2.imread('/home/william/code/ar10000/mountain_goat/raw_data/mountain_goat_screenshots/video106/Screenshot 2022-08-29 at 12.14.30.png')
    # # img5 = cv2.imread('/home/william/code/ar10000/mountain_goat/raw_data/mountain_goat_screenshots/video106/Screenshot 2022-08-29 at 12.14.42.png')
    # list_frames =[img1, img2, img3, img4]
    """takes in a list of frames and draws the next move """
    frame_1 = list_frames[-1]
    # frame_last = img5
    frame_coordinates = []
    for frame in list_frames:
        body = get_pose_image(frame)
        frame_coordinates.append(body)
    frame_coordinates_df = pd.DataFrame(frame_coordinates)
    shape = frame_coordinates_df.shape

    X= np.ones((25, 8))
    X= X*-1000
    X[:shape[0], :]= frame_coordinates_df.to_numpy(dtype='float32')
    X_pred = np.ones((1, 25, 8))
    X_pred = X_pred *-1000
    X_pred[0,:shape[0],:] = frame_coordinates_df.to_numpy(dtype='float32')
    #load model from storage
    model= keras.models.load_model(next_move_model)

    # check if something is in cloud
    if model is None:
        model , history = train()
        model.save('next_move_model')

    prediction = model.predict(X_pred)
    # ipdb.set_trace()
    image_dim = frame_1.shape
    left_hand = int(prediction[0, 0]* image_dim[0]), int(prediction[0, 1]*image_dim[1])
    right_hand = int(prediction[0, 2]* image_dim[0]), int(prediction[0, 3]*image_dim[1])
    left_foot = int(prediction[0, 4]* image_dim[0]), int(prediction[0, 5]*image_dim[1])
    right_foot = int(prediction[0, 6]* image_dim[0]), int(prediction[0, 7]*image_dim[1])
    # ipdb.set_trace()
    frame_1 = grip_colors(frame_1, grip_model )[0]
    cv2.circle(img=frame_1, center=tuple(left_hand), radius=100, color=(255, 0, 0), thickness=10)
    cv2.circle(img=frame_1, center=tuple(right_hand), radius=100, color=(255, 0, 0), thickness=10)
    cv2.circle(img=frame_1, center=tuple(left_foot), radius=100, color=(0, 255, 0), thickness=10)
    cv2.circle(img=frame_1, center=tuple(right_foot), radius=100, color=(0, 255, 0), thickness=10)
    # frame_1 = frame_1[...,::-1]

# Display the resulting frame
    fig = plt.imshow(frame_1)
    plt.title("Image with grips")
    plt.show()


    return frame_1






if __name__ == '__main__':
    # model, history = train()
    grip_model = 'raw_data/output/model_final.pth'
    next_move = 'next_move_model'
    print(next_position(grip_model, next_move))
