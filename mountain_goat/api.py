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



# http://127.0.0.1:8000/predict? \
    # pickup_datetime=2012-10-0612:10:20 \
        # &pickup_longitude=40.7614327 \
            # &pickup_latitude=-73.9798156 \
                # &dropoff_longitude=40.6513111 \
                    # &dropoff_latitude=-73.8803331 \
                        # &passenger_count=2


@app.get("/")
def root():
    return {'greeting': 'Hello'}
