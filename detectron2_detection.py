from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


class Detectron2:

    def __init__(self):
        # Create config file for detectron
        # TODO: ide valamit

        self.cfg = get_cfg()
        #self.cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        #self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        #self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        
        self.cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"



        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
        self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.1]
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
        self.cfg.MODEL.RPN.IOU_THRESHOLDS = [0.1, 0.4]

        self.cfg.MODEL.FPN.FPN_ON = True
        self.cfg.MODEL.FPN.MULTILEVEL_RPN = True
        self.cfg.MODEL.FPN.RPN_MIN_LEVEL = 1
        self.cfg.MODEL.FPN.RPN_MAX_LEVEL = 8
        self.cfg.MODEL.FPN.COARSEST_STRIDE = 256
        self.cfg.MODEL.FPN.SCALES_PER_OCTAVE = 3
        self.cfg.MODEL.FPN.ANCHOR_SCALE = 2



        # Initializes the predictor object with the config file
        # CUDA nélküli futtatáshoz
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)

    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax, rmax

    def detect(self, im):
        # Predictor performs detection here
        outputs = self.predictor(im)
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        
        bbox_xcycwh, cls_conf, cls_ids = [], [], []

        for (box, _class, score) in zip(boxes, classes, scores):
            # Select only person class for prediction
            if _class == 0:
                # predicted boxes contain the top left x0, y0, and the bottom right x1, y1
                # DeepSort needs bound box centers and Width and Height
                x0, y0, x1, y1 = box
                bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
                cls_conf.append(score)
                cls_ids.append(_class)

        return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids)
