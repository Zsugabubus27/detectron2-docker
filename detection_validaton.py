import argparse
import os
import time
from distutils.util import strtobool
import pandas as pd

import cv2
from detectron2_detection import Detectron2
from util import draw_bboxes
import pickle

class Detector(object):
    def __init__(self, args):
        self.args = args
        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()

    def __enter__(self):
        assert os.path.isfile(self.args.video_path), "Error: path error"
        self.vdo.open(self.args.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.output_path, fourcc, self.args.fps, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        # Check wheter there is next frame
        allDetection = dict()
        idx_frame = 0

        while self.vdo.grab():
            start = time.time()

            # Retrieve next frame
            _, im = self.vdo.retrieve()
            
            if idx_frame < self.args.start_frame:
                print('Skip frame', idx_frame)
                idx_frame += 1
                continue
            elif idx_frame > self.args.end_frame:
                break

            # Detect object on image
            bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)

            # TODO: Kell ide null check?  
            if bbox_xcycwh is not None: # and len(bbox_xcycwh) > 0
                # NOTE: This is double check since all the returned boxes are person objects (in the detect funcion it is asserted)
                # select class person
                mask = cls_ids == 0          
                cls_conf = cls_conf[mask]

                # NOTE: only the height is multiplies by 1.2, why?
                # ANSWER: bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                # TODO: Uncomment 1.1 
                bbox_xcycwh = bbox_xcycwh[mask]
                #bbox_xcycwh[:, 3:] *= 1.1

                # Detekciók kimentése, hogy megnézzem a KF-hez a mátrixokat.
                allDetection[idx_frame] = [bbox for bbox, conf in zip(bbox_xcycwh, cls_conf) if conf > self.args.min_confidence]
                
                idx_frame += 1

                if self.args.output_path:
                    # Összes box kirjazolása
                    bb_xyxy = [
                        [xc - w/2, yc - h / 2 , xc + w/2, yc + h / 2] 
                        for xc, yc, w, h in bbox_xcycwh]
                    bb_xyxy = [x for x, conf in zip(bb_xyxy, cls_conf) if conf > self.args.min_confidence]
                    all1 = [1]*len(bb_xyxy)
                    im = draw_bboxes(im, bb_xyxy, all1)
                    self.output.write(im)

            end = time.time()
            print("time: {}s, fps: {}, frame: {}".format(end - start, 1 / (end - start), idx_frame - 1))
        
        with open(self.args.detection_output, 'wb') as handle:
            pickle.dump(allDetection, handle, protocol=pickle.HIGHEST_PROTOCOL)
            


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--detection_output", type=str, default=None)
    parser.add_argument("--end_frame", type=int, default=1e10)
    parser.add_argument("--start_frame", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.video_path is None:
        print('Debugging...')
        args.output_path = "debug.avi"
        args.video_path = "/home/dobreff/work/Dipterv/MLSA20/data/ISSIA_SoccerDataset/Sequences/vendeg_elorol.avi"
        with Detector(args) as det:
            det.detect()
    else:
        with Detector(args) as det:
            det.detect()
