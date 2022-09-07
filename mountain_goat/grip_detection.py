from bdb import Breakpoint
from importlib.resources import path
from detectron2.utils.logger import setup_logger
setup_logger()
# from mountain_goat.params import BASE_DIR
import os, json, cv2, random
##import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
import ipdb
from PIL import Image

def train_detectron2():
    '''training detectron2'''
    if os.environ.get('DATA_SOURCE')== 'local':
        path_annotations= os.environ.get('LOCAL_PATH_GRIPS_ANNOTATIONS')
        path_grip_train= os.environ.get('LOCAL_PATH_GRIPS')
    elif os.environ.get('DATA_SOURCE')== 'cloud':
        pass
       #TODO DOWNLOAD DATA FROM CLOUD
        # path_annotations = f'{BASE_DIR}/raw_data/train/_annotations.coco.json'
        # path_grip_train = f'{BASE_DIR}/raw_data/train/'
    #registering dataset to detectron2
    register_coco_instances("train",{}, path_annotations, path_grip_train)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # /Base-RCNN-C4.yaml
    #TODO train path
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
    #TODO upload to cload
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def get_grips(image, model_path):
#     path_annotations = 'raw_data/mountain_goat_UCSD/hold_detection_dataset/train/_annotations.coco.json'
#     path_grip_train ='raw_data/mountain_goat_UCSD/hold_detection_dataset/train'

#     try :
#         train_metadata = MetadataCatalog.get("train")
#         DatasetCatalog.get("train")
#     except:
#         register_coco_instances("train",{}, path_annotations, path_grip_train)
#         train_metadata = MetadataCatalog.get("train")
#         DatasetCatalog.get("train")
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
#     # /Base-RCNN-C4.yaml
#     cfg.DATASETS.TRAIN  = ("train",)
#     cfg.DATASETS.TEST  = ()
#     cfg.DATALOADER.NUM_WORKERS = 2
#     cfg.MODEL.WEIGHTS =  os.path.join(model_path)
#     cfg.SOLVER.IMS_PER_BATCH =2
#     cfg.SOLVER.BASE_LR = 0.00025
#     cfg.SOLVER.MAX_ITER = 1600
#     cfg.SOLVER_STEPS = []
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
#     cfg.MODEL.DEVICE='cpu'

#     predictor = DefaultPredictor(cfg)

#     outputs = predictor(image)
#     cv2.imwrite("temp.jpeg", image)
#     im = cv2.imread("temp.jpeg")
#     v = Visualizer(im[:, :, ::-1],
#                      metadata=train_metadata,
#                      scale=1,
#                      instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     return out

# if __name__ == '__main__':
#     model_path = 'models_output/grip_detection/model_final.pth'
#     image_path = 'raw_data/mountain_goat_screenshots/video1/Screenshot (92).png'
#     im = cv2.imread(image_path)
#     print(get_grips(im, model_path))


    path_annotations = 'raw_data/mountain_goat_UCSD/hold_detection_dataset/train/_annotations.coco.json'
    path_grip_train ='raw_data/mountain_goat_UCSD/hold_detection_dataset/train'

    try :
        train_metadata = MetadataCatalog.get("train")
        DatasetCatalog.get("train")
    except:
        register_coco_instances("train",{}, path_annotations, path_grip_train)
        train_metadata = MetadataCatalog.get("train")
        DatasetCatalog.get("train")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # /Base-RCNN-C4.yaml
    cfg.DATASETS.TRAIN  = ("train",)
    cfg.DATASETS.TEST  = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS =  os.path.join(model_path)
    cfg.SOLVER.IMS_PER_BATCH =2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1600
    cfg.SOLVER_STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE='cpu'

    predictor = DefaultPredictor(cfg)

    outputs = predictor(image)
    cv2.imwrite("temp.jpeg", image)
    im = cv2.imread("temp.jpeg")
    v = Visualizer(im[:, :, ::-1],
                     metadata=train_metadata,
                     scale=1,
                     instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return outputs['instances'].pred_boxes.tensor.cpu().numpy()

if __name__ == '__main__':
    model_path = 'models_output/grip_detection/model_final.pth'
    image_path = 'raw_data/mountain_goat_screenshots/video1/Screenshot (92).png'
    im = cv2.imread(image_path)
    print(get_grips(im, model_path))
