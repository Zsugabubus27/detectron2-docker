import argparse
import os
import time
from distutils.util import strtobool
import pandas as pd

import cv2

from deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes, draw_dead_bboxes, draw_frameNum
from deep_sort import coord_mapper
import pickle

import natsort
import glob

class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))

        #self.vdo = cv2.VideoCapture()
        self.imgList = natsort.natsorted(glob.glob(self.args.imgs_path))
        self.detectron2 = Detectron2()

        # Initialize coordinate mapper
        self.myCoordMapper = coord_mapper.CoordMapperCSG(match_code='HUN-BEL 2. Half')
        self.fps = 6

        self.deepsort = DeepSort(args.deepsort_checkpoint, lambdaParam=0.6, coordMapper=self.myCoordMapper, max_dist=1.0, min_confidence=0.1, 
                        nms_max_overlap=0.7, max_iou_distance=0.7, max_age=self.fps*3, n_init=3, nn_budget=50, use_cuda=use_cuda)

    def __enter__(self):
        #assert os.path.isfile(self.args.video_path), "Error: path error"
        #self.vdo.open(self.args.video_path)
        #self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        #self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        img = cv2.imread(self.imgList[0])
        self.im_height, self.im_width, _ = img.shape


        # FIXME: Output FPS is hardcoded to 20
        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, self.fps, (self.im_width, self.im_height))

        #assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        # Check wheter there is next frame
        results = []
        allDetection = dict()
        idx_frame = 0


        #while self.vdo.grab():
        while idx_frame < len(self.imgList):
            start = time.time()

            # Retrieve next frame
            #_, im = self.vdo.retrieve()
            im = cv2.imread(self.imgList[idx_frame])
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # only for images

            # Detect object on image
            bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)
            detection_mask = [(xc, yc+(h/2)) for xc, yc, w, h in bbox_xcycwh]
            detection_mask = self.myCoordMapper.image2xy(detection_mask)
            detection_mask = [False if x is None else True for x in detection_mask]
            bbox_xcycwh, cls_conf, cls_ids = bbox_xcycwh[detection_mask], cls_conf[detection_mask], cls_ids[detection_mask]

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

                idx_frame += 1
                # Összes box kirjazolása
                bb_xyxy = [
                    [xc - w/2, yc - h / 2 , xc + w/2, yc + h / 2] 
                    for xc, yc, w, h in bbox_xcycwh]
                bb_xyxy = [x for x, conf in zip(bb_xyxy, cls_conf) if conf > self.deepsort.min_confidence]
                all1 = [None]*len(bb_xyxy)
                im = draw_bboxes(im, bb_xyxy, all1)

                # Do tracking
                outputs, deadtracks = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                print('len outputs:{0}, len deadtracks:{1}'.format(len(outputs), len(deadtracks)))
                
                # Draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    im = draw_bboxes(im, bbox_xyxy, identities)

                    # Write to file
                    bbox_tlwh = [self.deepsort._xyxy_to_tlwh(bb) for bb in bbox_xyxy]
                    results.append((idx_frame - 1, bbox_tlwh, identities))

                im = draw_frameNum(im, (2514, 330), idx_frame - 1)

                # Draw boxes for dead tracks for debugging
                if len(outputs) > 0:
                    bbox_xyxy = [x[:4] for x in deadtracks]
                    labels = [x[-1] for x in deadtracks]
                    im = draw_dead_bboxes(im, bbox_xyxy, labels)
                

            end = time.time()
            print("time: {}s, fps: {}, frame: {}".format(end - start, 1 / (end - start), idx_frame - 1), '\n', '-'*30, '\n')

            if self.args.save_path:
                self.output.write(im)

        # Write all tracked objs to file
        write_results(self.args.result_path, results, 'mot')
def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h}\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.video_path is None:
        print('Debugging...')
        args.use_cuda = "False"
        args.save_path = "/mnt/data/mlsa20_cr/out/HUN_BEL_second_half.avi"
        args.result_path = "/mnt/data/mlsa20_cr/out/HUN_BEL_second_half.txt"
        #args.video_path = "/home/dobreff/work/Dipterv/MLSA20/data/video_46000_47000.avi"
        args.imgs_path = "/mnt/data/mlsa20_cr/src/masodik_felido/*.png"
        with Detector(args) as det:
            det.detect()
    else:
        with Detector(args) as det:
            det.detect()
