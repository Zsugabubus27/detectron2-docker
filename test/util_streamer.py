import cv2
from abc import ABC, abstractmethod
import os
import natsort
import numpy as np
from queue import Queue
from threading import Thread
import time
import zmq
import imagezmq
import sys

class ImagePublisher():
	def __init__(self, connect_to='tcp://*:9998'):
		socketType = zmq.PUB
		self.zmq_context = imagezmq.SerializingContext()
		self.zmq_socket = self.zmq_context.socket(socketType)
		self.zmq_socket.bind(connect_to)

	def send_image(self, msg, img : np.ndarray):
		

		compressed_array = img
		md = dict(
			msg=msg,
			dtype=str(img.dtype),
			shape=img.shape,
		)
		self.zmq_socket.send_json(md, 0 | zmq.SNDMORE)
		return self.zmq_socket.send(compressed_array, 0, copy=True, track=False)



class Undistorter():
	'''
	This class is responsible for undistorting an image by giving the K and D matrices.
	Important!
	K and D have to be calculated with cv2.fisheye.calibrate
	It is a camera specific value.
	'''
	def __init__(self, _K, _D, DIM, dim1, outputDim, dim2=None, dim3=None, balance=1.5):
		self.K = _K
		self.D = _D
		self.outputDim = outputDim
		self.inputDim = dim1
		self.resizeNeeded = (self.outputDim != self.inputDim)
		self.mat1, self.mat2 = self._preprocess(DIM=DIM, balance=balance, dim1=dim1, dim2=dim2, dim3=dim3)

	def _preprocess(self, DIM, dim1, dim2, dim3, balance ):
		# dim1 is the dimension of input image to un-distort (width, height)
		# assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same 
		# 											aspect ratio as the ones used in calibration"
		dim1 = np.array(dim1).astype(int)
		dim2 = dim2 or dim1
		dim3 = dim3 or dim1

		# The values of K is to scale with image dimension.
		# Except that K[2][2] is always 1.0	
		# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. 
		# OpenCV document failed to make this clear!
		scaled_K = self.K * dim1[0] / DIM[0]
		scaled_K[2][2] = 1.0

		new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.D, tuple(dim2), np.eye(3), balance=balance)
		map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, tuple(dim3), cv2.CV_16SC2)
		return map1,map2

	def undistort(self, img):
		'''
		Undistorts the given image.
		'''
		undistorted_img = cv2.remap(img, self.mat1, self.mat2, interpolation=cv2.INTER_LINEAR, 
									borderMode=cv2.BORDER_CONSTANT)

		if self.resizeNeeded:
			return cv2.resize(undistorted_img, self.outputDim, interpolation = cv2.INTER_AREA)
		else:
			return undistorted_img 



class AbstractCamera(ABC):
	def __init__(self):
		super().__init__()
	
	@abstractmethod
	def getNextFrame(self):
		pass
	
	@abstractmethod
	def getDimension(self):
		'''
		Return the dimension (width, height)
		'''
		pass
	
	@abstractmethod
	def forwardByOffset(self, offset):
		pass

class Camera(AbstractCamera):
	def __init__(self, folderPath, startTime, videoExtension='MP4'):
		'''
		params:
			folderPath = folder which contains the videos named like: 153306AA.MP4
			startTime = start of the match in format HHMMSS like 153121
		'''
		startTime = int(startTime)
		listVideos = [vid for vid in os.listdir(folderPath) if videoExtension.lower() in vid.lower()]
		# TODO: Mi van ha a fájl 150000 és a meccs 150010-kor keződid?
		listVideos = [vid for vid in listVideos if int(vid[:6]) > startTime]
		listVideos = natsort.natsorted(listVideos)
		listVideos = [os.path.join(folderPath, vid) for vid in listVideos]
		self.videos = iter(listVideos)
		self.vdo = cv2.VideoCapture(next(self.videos))

	def getDimension(self):
		'''
		Return the dimension (width, height)
		'''
		height = self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT)
		width = self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH)
		return width, height
	def getNextFrame(self):
		ret, frame = self.vdo.read()
		if not ret:
			self.vdo = cv2.VideoCapture(next(self.videos))
			return self.getNextFrame()
		return frame
	def forwardByOffset(self, offset):
		for i in range(offset):
			self.getNextFrame()

class CameraStream(AbstractCamera):
	def __init__(self, folderPath, startTime, videoExtension='MP4', transform=None, queue_size=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		startTime = int(startTime)
		listVideos = [vid for vid in os.listdir(folderPath) if videoExtension.lower() in vid.lower()]
		# TODO: Mi van ha a fájl 150000 és a meccs 150010-kor keződid?
		listVideos = [vid for vid in listVideos if int(vid[:6]) > startTime]
		listVideos = natsort.natsorted(listVideos)
		listVideos = [os.path.join(folderPath, vid) for vid in listVideos]
		self.videos = iter(listVideos)
		self.stream = cv2.VideoCapture(next(self.videos))
		self.stopped = False
		self.transform = transform

		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queue_size)
		# intialize thread
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True

	def forwardByOffset(self, offset : int):
		'''
		Forwards the video by the given frame offset
		When the forwarded offset higher than the length of the video jumps to the next video file.
		'''
		# Video hossza framekben, ha len = 50, akkor az a 49.frame
		lenVideo = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
		currPos = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
		desiredPos = currPos + offset 

		if desiredPos < lenVideo:
			self.stream.set(cv2.CAP_PROP_POS_FRAMES, desiredPos)
		else:
			# 1. Jump to the next videofile
			self.stream = cv2.VideoCapture(next(self.videos))
			# 2. Offset = desiredPosition - lengthOfVideo
			self.forwardByOffset(desiredPos - lenVideo)

	def getDimension(self):
		'''
		Return the dimension (width, height)
		'''
		height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
		width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
		return width, height

	def setTransform(self, transform):
		self.transform = transform

	def start(self):
		# start a thread to read frames from the file video stream
		self.thread.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				break

			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()

				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				# 1: We check whether there is a next video file
				if not grabbed:
					try:
						self.stream = cv2.VideoCapture(next(self.videos))
						(grabbed, frame) = self.stream.read()
					except StopIteration:
						# If there is no more video files in the folder
						self.stopped = True
					
				# if there are transforms to be done, might as well
				# do them on producer thread before handing back to
				# consumer thread. ie. Usually the producer is so far
				# ahead of consumer that we have time to spare.
				#
				# Python is not parallel but the transform operations
				# are usually OpenCV native so release the GIL.
				#
				# Really just trying to avoid spinning up additional
				# native threads and overheads of additional
				# producer/consumer queues since this one was generally
				# idle grabbing frames.
				if self.transform:
					frame = self.transform(frame)

				# add the frame to the queue
				self.Q.put(frame)
			else:
				time.sleep(0.1)  # Rest for 10ms, we have a full queue

		self.stream.release()

	def getNextFrame(self):
		return self.read()

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	# Insufficient to have consumer use while(more()) which does
	# not take into account if the producer has reached end of
	# file stream.
	def running(self):
		return self.more() or not self.stopped

	def more(self):
		# return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
		tries = 0
		while self.Q.qsize() == 0 and not self.stopped and tries < 1000000:
			time.sleep(0.05)
			tries += 1

		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		# wait until stream resources are released (producer thread might be still grabbing frame)
		self.thread.join()

