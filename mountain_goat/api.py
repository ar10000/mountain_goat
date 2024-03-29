import io
import cv2
import numpy as np
import base64
import shutil
import emoji
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from mountain_goat import next_move_model
from mountain_goat.grip_detection import get_grips
from mountain_goat.color_identification import grip_colors
from mountain_goat.main import next_position
from mountain_goat.significant_frames import get_significant_frames


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins ?
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    )

@app.get("/")
def root():
    return {'Home':'Welcome to home page'}



@app.post("/pred_move")
def test(list_frames_1: bytes = File(default = None),list_frames_2: bytes = File(default = None)
         ,list_frames_3: bytes = File(default = None), list_frames_4: bytes = File(default = None),list_frames_5: bytes = File(default = None),
         list_frames_6: bytes = File(default = None),list_frames_7: bytes = File(default = None)):

    print(1)
    list_array_images =[]
    listos = []
    # receiving list of bytes , decoding and transforming to array
    # img in list_frames:
    #for img in list_frames:
    print(2)
    res = [list_frames_1, list_frames_2, list_frames_3, list_frames_4,list_frames_5,  list_frames_6, list_frames_7]
    print(f"Length of the images before decoding : {len(res)}")
    for img in res:
        if img is not None:
            listos.append(img)


    for list_frames in listos:
        file_decoded = base64.decodebytes(list_frames)#decoding and saving as file in memory
        # Opening the file in memory, transforming it as an array
        print(3)
        image_arr = np.asarray(bytearray(file_decoded), dtype = "uint8")
        # print(type(image_arr))
        # print(image_arr.shape)
        print(4)
        image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        list_array_images.append(image)
        #run prediction
    print(f"Length of the images after decoding : {len(list_array_images)}")
    print(5)
    grip_model_path = 'raw_data/output/model_final.pth'
    print(6)
    next_move_model = 'next_move_model'
    print(7)
    prediction = next_position(grip_model_path, next_move_model, list_array_images)
    print(8)
    # We save and open the output from get_grips
    cv2.imwrite("temp2.jpeg", prediction)
    print(9)
    new_image = Image.open("temp2.jpeg")
    print(10)

    # Code to send back the picture to front-end with
    new_image_file = io.BytesIO()   # Creating a new empty file in memory to store the image
    new_image.save(new_image_file, "JPEG")   # Saving the rotated image to the new file in memory
    new_image_file.seek(0)          # Go to the start of the file before starting to send it as the response
    return StreamingResponse(new_image_file, media_type='image/jpeg') # Sending the response


def save_file_tmp(file) -> Path :
    """takes in a file and saves it temporarily and
    returns the path of the file
    """
    suffix = Path(file.filename).suffix
    try:
         with NamedTemporaryFile(delete=False, suffix=suffix) as temp_vid_file:
            # copy file recieved and store it temporarily
            shutil.copyfileobj(file.file, temp_vid_file)
            # get path of stored
            temp_vid_path = Path(temp_vid_file.name)
    finally:
        file.file.close()
    return temp_vid_path


@app.post('/pred_move_2')
def upload_video(file: UploadFile):
    file_path = save_file_tmp(file)
    try:
        significant_frames = get_significant_frames(str(file_path))
        print(f'#######number of significant frames found is {len(significant_frames)}')
        print(f'##### items in significant frames have {significant_frames[0].shape} shape')
    finally:
        file_path.unlink()

    print('############# loading models  ################')
    #load grip model path
    grip_model_path = 'raw_data/output/model_final.pth'

    #load next_move_model_path
    next_move_model = 'next_move_model'

    #making a prediction
    print(emoji.emojize('################ making predictions :upside-down_face:...... '))
    prediction = next_position(grip_model_path, next_move_model, significant_frames)

    print(f'######## Done the prediction is a {type(prediction)}###########')

    # returning image
    cv2.imwrite("temp2.jpeg", prediction)
    prediction_image = Image.open("temp2.jpeg")

    # Code to send back the picture to front-end with
    new_image_file = io.BytesIO()   # Creating a new empty file in memory to store the image
    prediction_image.save(new_image_file, "JPEG")   # Saving the rotated image to the new file in memory
    # new_image_file.seek(0)          # Go to the start of the file before starting to send it as the response
    return StreamingResponse(new_image_file, media_type='image/jpeg') # Sending the response
