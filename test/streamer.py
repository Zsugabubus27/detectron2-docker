import zmq
import imagezmq
import cv2
import os
import numpy as np
import time # DEBUG
from util_streamer import Camera, Undistorter, CameraStream, ImagePublisher
import pickle
from detectron_client import DetectronClient
from utils import draw_bboxes
from queue import Queue


# ---- INPUTS ---------------------------------------------------
# Parameters for undistortion
DIM=(2560, 1440)
K= np.array([[1421.252328579295, 0.0, 1284.3399064769028], [0.0, 1421.5991651379297, 691.2988521557191], [0.0, 0.0, 1.0]])
D= np.array([[-0.04903868008658097], [-0.08844440260190703], [0.228408663794075], [-0.1822240688394524]])

# Original params of video
videoFPS = 60
origResolution = (2560, 1440) # Width, Height

# Desired params of output stream
outputFPS = 30
outResolution = (2560, 1440) # Width, Height

DEBUG_LEN_VIDEO = 900 # 60FPS-ben mért framek száma 36000
VIDEO_START_OFFSET = 0 # 60FPS-ben mért framek száma
# ---------------------------------------------------------------
NUMBER_OF_FRAMES =  DEBUG_LEN_VIDEO

# Init cameras:
# 1. Calculate FPS skipping
if (videoFPS % outputFPS) != 0:
	raise ValueError
stepFrame = int(videoFPS / outputFPS)
# 2. Calculate resizing images
# FIXME: Resizing is after undistorting the image
# Reason is because it is working now... Do not want to fuck it up....

# FIXME: Only for debugging, must be a smarter solution!!!
# '/home/dobreff/videos/first/190401000_1/' or '/mnt/match_videos/2019.03.29./190401000_1/'
left_camera_path = '/home/dobreff/videos/first/190401000_1/'
right_camera_path = '/home/dobreff/videos/first/190401000_2/'

StartTime = 153000 # begintime hhmmss - kiszopkodható a CTR GPS-ből ezzel most nem tökölök
leftCamera = CameraStream(left_camera_path, StartTime, 'MP4')
rightCamera = CameraStream(right_camera_path, StartTime, 'MP4')
# TODO: Prepare for N number of cameras. The cameras should be in a list.

# Sync the cameras
# Steps:
# 1. Megneztem a videofelveteleken hogy mikor van sipszo es odatekertem
# 2. Aztan kigeneraltam a képeket
# Es proba alapon meg az egyik kamerat tovabb tekertem hogy pontos legyen
# HUN_BEL 1. felido:
# 	bal kamera: 153005AA-val kezdodik, 1:00-nál van sip
# 	jobb kamera: 153018AA-val kezdodik, 1:54-nál van sip + 20 frame offset

# Delta sync
rightCamera.forwardByOffset(54*videoFPS + 20)

# Whistle forward
whistle = 60*videoFPS
leftCamera.forwardByOffset(whistle)
rightCamera.forwardByOffset(whistle)
# Forward video by given framecount
leftCamera.forwardByOffset(VIDEO_START_OFFSET)
rightCamera.forwardByOffset(VIDEO_START_OFFSET)

# Init the Undistorter
imgDim = origResolution
undistorter = Undistorter(K, D, DIM=DIM, dim1=imgDim, outputDim=outResolution, balance=1.5)

leftCamera.setTransform(undistorter.undistort)
rightCamera.setTransform(undistorter.undistort)

leftCamera.start()
rightCamera.start()
time.sleep(1.0)
print('Frame reading started!')

# ---------------------------------- Init Consumers aka. Detectron clients -----------------------
# Queue from which the consumers grab the frames
queue_frames = Queue(maxsize=128)

# Result Queue where the results are appended
queue_results = Queue()

# Initialize Detectron Clients
dc1 = DetectronClient('localhost:5555', queue_frames, queue_results, verbose=False)
dc2 = DetectronClient('localhost:5556', queue_frames, queue_results, verbose=False)
dc3 = DetectronClient('localhost:5557', queue_frames, queue_results, verbose=False)
dc4 = DetectronClient('localhost:5558', queue_frames, queue_results, verbose=False)

# Start processing
dc1.start()
dc2.start()
dc3.start()
dc4.start()

# ------------------------------------------------------------------------
list_times = []
debugTimes=[]
sendTimes = []

frameNum = 0
while leftCamera.more() and rightCamera.more():
	# 0. Skip frames to maintain desired FPS
	if (frameNum % stepFrame) != 0:
		frameNum +=1
		continue

	ts_start = time.time()
	# TODO: Handle when it is the end of the video. Camera raises StopIteration Error!!!
	# 1. Get next frame for each camera
	# 2. Undistort the frame for each frame -> This happends in the camera
	leftCameraFrame = leftCamera.getNextFrame()
	rightCameraFrame = rightCamera.getNextFrame()
	
	ts_frames_received = time.time()
	
	# 3. Concat the frames horizontally
	frame = cv2.hconcat([leftCameraFrame, rightCameraFrame])
	ts_concat = time.time()

	# 4. Put the frame into the input Queue
	queue_frames.put( {'idx' : frameNum, 'image' : frame} )
	
	ts_send_done = time.time()

	frameNum +=1
	
	# list_times.append({'frameNum_streamer' : frameNum, 'ts_start' : ts_start, 'ts_frames_received' : ts_frames_received, 
	# 					'ts_concat' : ts_concat, 'ts_send_done' : ts_send_done})
	
	# sendTimes.append(ts_send_done - ts_concat)
	# debugTimes.append(ts_send_done - ts_start)
	# print(frameNum, ts_send_done - ts_start, sum(debugTimes) / len(debugTimes))
	# if (outputFPS == 6) and int(frameNum) >= 4000:
	# 	time.sleep(0.1)

	if int(frameNum) >= (NUMBER_OF_FRAMES):
		print('Every frame has been read in!')
		break

leftCamera.stop()
rightCamera.stop()

# Every frame has been put into the queue_frame
# We are waiting to be processed
# Wait to process every frame:
while queue_results.qsize() < (NUMBER_OF_FRAMES // stepFrame):
	print(queue_results.qsize(), queue_frames.qsize())
	time.sleep(1)

# Save results
dict_results = {}
while (not queue_results.empty()):
	# TODO: imaget nem teszem el élesben
	idx, img_orig, list_res = queue_results.get()
	dict_results[idx] = list_res
	newImg = draw_bboxes(img_orig, list_res)
	cv2.imwrite(f'/home/dobreff/videos/outputs/{outResolution[0]}_{outputFPS}fps/{idx}.jpg', newImg)
	print(f'Item {idx} saved!')

with open(f'/home/dobreff/videos/outputs/{outResolution[0]}_{outputFPS}fps/detections.pickle', 'wb') as handle:
	pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # TODO: Debug --------------------------------------------------------------------------------------------------
# # Wait until every frame is sent
# print('Wait until every ')
# streamWriter.wait_until_empty()

# dfTimes = pd.concat([pd.DataFrame(list_times), pd.DataFrame(streamWriter.timeList)], axis=1)
# dfTimes.to_csv('streamer.csv')
# # Then for every row i am writing out the times
# # for a, b in zip(list_logger, streamWriter.timeList):
# # 	myLogger.write(*(a+b))

# print('Vegeztem! Orokke varok!')
# while True:
# 	time.sleep(20)


	
