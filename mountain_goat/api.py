from datetime import datetime
#from imp import load_compiled
#from json import load
#from unittest.util import strclass
import pandas as pd
import pytz
#from urllib import request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    return {'greeting': 'Hello'}

@app.get("/predict")
def root():
    #RECEIVE THE IMAGE FROM THE FRONT END

    #PROCESS THE IMAGE INTO A NP ARRAY

    #LOAD MODEL

    #PREDICT
    return {'colour': 'red'}
