import argparse
import os
import time
from distutils.util import strtobool

import cv2

from deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()
        # FIXME: Max dist itt nem szerepelt
        self.deepsort = DeepSort(args.deepsort_checkpoint, lambdaParam=0.6, max_dist=1.0, min_confidence=0.1, 
                        nms_max_overlap=0.7, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FIXME: Output FPS is hardcoded to 20
        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 6, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        # Check wheter there is next frame
        results = []
        idx_frame = 0
        while self.vdo.grab():
            start = time.time()
            idx_frame += 1

            # Retrieve next frame
            _, im = self.vdo.retrieve()
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # Detect object on image
            bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)

            # TODO: Kell ide null check?  
            if bbox_xcycwh is not None: # and len(bbox_xcycwh) > 0
                # FIXME: This is double check since all the returned boxes are person objects (in the detect funcion it is asserted)
                # select class person
                mask = cls_ids == 0          
                cls_conf = cls_conf[mask]

                # FIXME: only the height is multiplies by 1.2, why?
                # ANSWER: bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.1

                # Összes box kirjazolása
                bb_xyxy = [
                    [xc - w/2, yc - h / 2 , xc + w/2, yc + h / 2] 
                    for xc, yc, w, h in bbox_xcycwh]
                bb_xyxy = [x for x, conf in zip(bb_xyxy, cls_conf) if conf > self.deepsort.min_confidence]
                all1 = [1]*len(bb_xyxy)
                im = draw_bboxes(im, bb_xyxy, all1)



                # # Do tracking
                # outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                
                # # Draw boxes for visualization
                # if len(outputs) > 0:
                    
                #     bbox_xyxy = outputs[:, :4]
                #     identities = outputs[:, -1]
                #     im = draw_bboxes(im, bbox_xyxy, identities)

                #     # Write to file
                #     bbox_tlwh = [self.deepsort._xyxy_to_tlwh(bb) for bb in bbox_xyxy]
                #     results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()
            print("time: {}s, fps: {}, frame: {}".format(end - start, 1 / (end - start), idx_frame - 1))
            if self.args.display:
                cv2.imshow("test", im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(im)

        # Write all tracked objs to file
        write_results("results.txt", results, 'mot')
            
        # exit(0)

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
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
