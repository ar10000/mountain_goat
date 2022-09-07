import io
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
from mountain_goat.grip_detection import get_grips
import cv2
import numpy as np

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

@app.get("/predict_grip_detection")
def predict_grip_detection(img_bytes):
    #RECEIVE THE IMAGE FROM THE FRONT END
    print(img_bytes)
    #PROCESS THE IMAGE INTO A NP ARRAY
    # print(image)
    #LOAD MODEL

    #PREDICT
    return {'colour': img_bytes}

@app.post("/test")
def test(file: bytes = File(...)):
    # # Receiving the bytes, decoding and saving as file in memory
    # file_decoded = base64.decodebytes(file)
    # image_file = io.BytesIO(file_decoded) # File is in memory >> No need to "open" it

    # # Opening the file in memory as an image and transforming it

    # # USE CV2.IMREAD INSTEAD OF PIL

    # image = Image.open(image_file)  # Opening the file as an image
    # new_image = image.rotate(90)    # Creating a new rotated image
    # #new_image = get_grips(image, ....)   # Careful: image is an image here, not a path to an image
    # model_path = 'models_output/grip_detection/model_final.pth'
    # prediction = get_grips(image, model_path)
    # print(prediction)

    # # Saving the rotated image and prepare to send
    # new_image_file = io.BytesIO()   # Creating a new empty file in memory to store the image
    # new_image.save(new_image_file, "JPEG")   # Saving the rotated image to the new file in memory
    # new_image_file.seek(0)          # Go to the start of the file before starting to send it as the response
    # return StreamingResponse(new_image_file, media_type='image/jpeg')  # Sending the response


    # Receiving the bytes, decoding and saving as file in memory
    file_decoded = base64.decodebytes(file)
    #image_file = io.BytesIO(file_decoded) # File is in memory >> No need to "open" it

    # Opening the file in memory as an image and transforming it

    # USE CV2.IMREAD INSTEAD OF PIL
    image_arr = np.asarray(bytearray(file_decoded), dtype = "uint8")
    image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    #image = Image.open(image_file)  # Opening the file as an image
    #new_image = image.rotate(90)    # Creating a new rotated image
    #new_image = get_grips(image, ....)   # Careful: image is an image here, not a path to an image
    model_path = 'models_output/grip_detection/model_final.pth'
    prediction = get_grips(image, model_path)
    print(prediction)

    # Saving the rotated image and prepare to send
    new_image_file = io.BytesIO()   # Creating a new empty file in memory to store the image
    new_image = save(new_image_file, "JPEG")   # Saving the rotated image to the new file in memory
    new_image_file.seek(0)          # Go to the start of the file before starting to send it as the response
    return StreamingResponse(new_image_file, media_type='image/jpeg')  # Sending the response
