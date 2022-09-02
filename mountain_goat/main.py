import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mountain_goat.preprocessing import create_dataframe
from mountain_goat.next_move_model import initialize_model, compile_model, train_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ipdb

list_videos= create_dataframe('/home/william/code/ar10000/mountain_goat/raw_data/Videos')

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
        # ipdb.set_trace()
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

def train_next_move():
    """train model """

    if os.environ.get('DATA_SOURCE') == 'local':
        #if data is locally stored get it here
        d_path = os.environ.get('LOCAL_PATH_CLIMB')

    list_videos = create_dataframe(d_path)
    X_pad_train, X_pad_test, y_train, y_test = preprocess(list_videos)
    model = initialize_model()
    model = compile_model(model)
    model, history = train_model(model, X_pad_train, y_train)
#TODO return history somewhere else
    return model
