FROM python:3.8.12-buster
COPY requirements.txt /requirements.txt
COPY mountain_goat /mountain_goat
COPY next_move_model /next_move_model
COPY raw_data /raw_data
COPY train /train
COPY credentials.json /credentials.json
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/facebookresearch/detectron2.git
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD uvicorn mountain_goat.api:app --host 0.0.0.0 --port $PORT
