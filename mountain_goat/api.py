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
# from androguard.core.bytecodes.apk import APK
#from taxifare.interface.main import pred

#from taxifare.ml_logic.registry import load_model
#from taxifare.ml_logic.preprocessor import preprocess_features




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
    file_decoded = base64.decodebytes(file)
    contents = file_decoded
    print(len(contents))
    im = Image.open(io.BytesIO(contents))
    im2 = im.rotate(90)
    # im2.show()


    # file2 = base64.encodebytes(im2)
    # bytes_data = file2.getvalue()
    # print(type(bytes_data))
    new_img = io.BytesIO()
    im2.save(new_img, "JPEG")
    new_img.seek(0)

    return StreamingResponse(new_img, media_type='image/jpeg')
    # return {"result": bytes_data}
