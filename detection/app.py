import time
import cv2
import imagezmq
import numpy as np
import coord_mapper
from detection import PlayerDetectorDetectron2
import pandas as pd
import os

def initDetectronModel(img_width):
	resolutionDict = {2560 : (2560, 1440), 2048 : (2048, 1152), 
						1920 : (1920, 1080), 1536 : (1536, 864), 1280 : (1280, 720)}

	# Detectron object
	origRes = (2560, 1440)
	#outRes = (2560, 1440) #  ["(2560, 1440)", "(2048, 1152)", "(1920, 1080)", "(1536, 864)", "(1280, 720)"] 
	outRes = resolutionDict[img_width]

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
	return myDetector


# Store state of the actual Detectron
actDetectron = None
actWidth = None
fileName = None

image_hub = imagezmq.ImageHub(open_port='tcp://*:5555')
while True: 
	msg, image = image_hub.recv_image()
	
	img_width = int(msg.split('_')[0])
	idx = int(msg.split('_')[1])
	fileName = fileName if fileName is not None else idx

	if actWidth is None or img_width != actWidth:
		# Create new _detectron model
		del actDetectron
		actDetectron = initDetectronModel(img_width)
		actWidth = img_width



	startTs = time.time()
	
	list_detections = actDetectron.detectPlayersOnFrame(image)
	endTs = time.time()
	
	image_hub.zmq_socket.send_pyobj(list_detections)
	sendTs = time.time()

	dict_times = {'frameName' : idx, 'resolution' : actWidth, 'tsStart' : startTs, 'tsWorkDone' : endTs, 'tsSendDone' : sendTs}
	
	pd.DataFrame([dict_times]).to_csv(f'/tmp/{fileName}.csv', mode='a', header=(not os.path.exists(f'/tmp/{fileName}.csv')), index=None)