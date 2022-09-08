import io
import cv2
import numpy as np
import base64
from datetime import datetime
#from imp import load_compiled
#from json import load
#from unittest.util import strclass
import pandas as pd
import pytz
#from urllib import request
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from mountain_goat import next_move_model
from mountain_goat.grip_detection import get_grips
from mountain_goat.color_identification import grip_colors
from mountain_goat.main import next_position


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
    return {'greeting': 'Hello again'}


@app.post("/grip_detection")
def test(file: bytes = File(...)):
    # Receiving the bytes, decoding and saving as file in memory
    file_decoded = base64.decodebytes(file)

    # Opening the file in memory, transforming it as an array
    image_arr = np.asarray(bytearray(file_decoded), dtype = "uint8")
    # Decoding the array to transform it in a cv2 object
    image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

    # Run the prediction for grip_detection
    model_path = 'models_output/grip_detection/model_final.pth'
    prediction = get_grips(image, model_path)

    # We save and open the output from get_grips
    cv2.imwrite("temp2.jpeg", prediction.get_image()[:, :, ::-1])
    new_image = Image.open("temp2.jpeg")

    # Code to send back the picture to front-end with
    new_image_file = io.BytesIO()   # Creating a new empty file in memory to store the image
    new_image.save(new_image_file, "JPEG")   # Saving the rotated image to the new file in memory
    new_image_file.seek(0)          # Go to the start of the file before starting to send it as the response
    return StreamingResponse(new_image_file, media_type='image/jpeg') # Sending the response

@app.post("/colour_grip_detection")
def test(file: bytes = File(...)):
    # Receiving the bytes, decoding and saving as file in memory
    file_decoded = base64.decodebytes(file)

    # Opening the file in memory, transforming it as an array
    image_arr = np.asarray(bytearray(file_decoded), dtype = "uint8")
    # Decoding the array to transform it in a cv2 object
    image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

    # Run the prediction for grip_detection
    model_path = 'models_output/grip_detection/model_final.pth'
    prediction = grip_colors(image, model_path)[0]

    # We save and open the output from get_grips
    cv2.imwrite("temp2.jpeg", prediction)
    new_image = Image.open("temp2.jpeg")

    # Code to send back the picture to front-end with
    new_image_file = io.BytesIO()   # Creating a new empty file in memory to store the image
    new_image.save(new_image_file, "JPEG")   # Saving the rotated image to the new file in memory
    new_image_file.seek(0)          # Go to the start of the file before starting to send it as the response
    return StreamingResponse(new_image_file, media_type='image/jpeg') # Sending the response

@app.post("/pred_move")
def test(list_frames: bytes = File(...)):
    list_array_images =[]
    # receiving list of bytes , decoding and transforming to array
    for frame in list_frames:
        file_decoded = base64.decodebytes(frame)#decoding and saving as file in memory
        # Opening the file in memory, transforming it as an array
        image_arr = np.asarray(bytearray(file_decoded), dtype = "uint8")
        list_array_images.append(image_arr)
        #run prediction
    grip_model_path = 'raw_data/output/model_final.pth'
    next_move_model = 'next_move_model'
    prediction = next_position(grip_model_path, next_move_model, list_array_images)

    # We save and open the output from get_grips
    cv2.imwrite("temp2.jpeg", prediction)
    new_image = Image.open("temp2.jpeg")

    # Code to send back the picture to front-end with
    new_image_file = io.BytesIO()   # Creating a new empty file in memory to store the image
    new_image.save(new_image_file, "JPEG")   # Saving the rotated image to the new file in memory
    new_image_file.seek(0)          # Go to the start of the file before starting to send it as the response
    return StreamingResponse(new_image_file, media_type='image/jpeg') # Sending the response
