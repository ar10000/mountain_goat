FROM python:3.8.12-buster

COPY mountain_goat /mountain_goat
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/facebookresearch/detectron2.git

CMD uvicorn mountain_goat.api:app --host 0.0.0.0 --port $PORT
