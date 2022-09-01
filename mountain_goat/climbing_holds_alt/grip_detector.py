import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

#import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

#registering dataset to detectron2
from detectron2.data.datasets import register_coco_instances
register_coco_instances("train",{}, \
    '/home/william/code/ar10000/mountain_goat/raw_data/train/_annotations.coco.json', \
        "/home/william/code/ar10000/mountain_goat/raw_data/train")
##Training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(\
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# /Base-RCNN-C4.yaml
cfg.DATASETS.TRAIN  = ("train",)
cfg.DATASETS.TEST  = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH =2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1600
cfg.SOLVER_STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

# set a custom testing threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)
train_metadata = MetadataCatalog.get("train")
DatasetCatalog.get("train")
im = cv2.imread('/home/william/code/ar10000/mountain_goat/ArticleImageHandler.jfif')
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
                metadata=train_metadata,
                scale=0.5,
                # Remove the colors of unsegmented pixels.
                #This option is only available for segmentation models
                instance_mode=ColorMode.IMAGE_BW
)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# type(out)
cv2.imshow('climbnet', out.get_image()[:, :, ::-1])
