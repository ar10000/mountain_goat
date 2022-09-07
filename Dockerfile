FROM python:3.8.12-buster

COPY requirements.txt /requirements.txt
COPY mountain_goat /mountain_goat
COPY models_output /models_output

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/facebookresearch/detectron2.git

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD uvicorn mountain_goat.api:app --host 0.0.0.0 --port $PORT
