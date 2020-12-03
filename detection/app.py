import time
import cv2
import imagezmq
import numpy as np
import coord_mapper
from detection import PlayerDetectorDetectron2

# Detectron object
origRes = (2560, 1440)
outRes = (2560, 1440) #  ["(2560, 1440)", "(2048, 1152)", "(1920, 1080)", "(1536, 864)", "(1280, 720)"] 

left_side_corner_pixels =  [[1264.7023,499.5309],[1870.1519,549.1178 ],[2517.0312,1275.4386 ],[ 192.16057,548.8432 ]]
right_side_corner_pixels = [[755.81616,474.7149],[1361.8402,456.584],[2482.6863,568.66864],[257.55048,1084.8362]]
full_width = origRes[0]*2
right_side_corner_pixels_added_half_field = np.array([[p[0] + full_width/2, p[1]] for p in right_side_corner_pixels])

# CoordMapper a pálya széleivel inicializálva
coordMapper = coord_mapper.CoordMapperCSG(match_code=(left_side_corner_pixels, right_side_corner_pixels))

myDetector = PlayerDetectorDetectron2(leftSideCorners=left_side_corner_pixels, 
										rightSideCorners=right_side_corner_pixels_added_half_field, 
										coordMapper=coordMapper, origResolution=origRes, 
										outResolution=outRes, segnumx=2, segnumy=2, nmsThreshold=0.5)



image_hub = imagezmq.ImageHub(open_port='tcp://*:5555')
while True: 
	rpi_name, image = image_hub.recv_image()
	st = time.time()
	list_detections = myDetector.detectPlayersOnFrame(image)
	print('Full detection', time.time() - st)
	image_hub.zmq_socket.send_pyobj(list_detections)