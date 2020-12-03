import time
import numpy as np
from detectron_client import DetectronClient
import cv2
import imagezmq

class DetectionImageSender(imagezmq.ImageSender):
	def send_image_reqrep(self, msg, image):
		"""Sends OpenCV image and msg to hub computer in REQ/REP mode
		Arguments:
		  msg: text message or image name.
		  image: OpenCV image to send to hub.
		Returns:
		  A text reply from hub.
		"""
		if image.flags['C_CONTIGUOUS']:
			# if image is already contiguous in memory just send it
			self.zmq_socket.send_array(image, msg, copy=False)
		else:
			# else make it contiguous before sending
			image = np.ascontiguousarray(image)
			self.zmq_socket.send_array(image, msg, copy=False)
		hub_reply = self.zmq_socket.recv_pyobj()  # receive the reply message
		return hub_reply


def draw_bboxes(img, list_detection):
	# list_detection : list of dicts (x, y, w, h, color, x_world, y_world)
	# resizeFactor = outResolution[0] / origResolution[0]
	for detection in list_detection:
		(x1,y1),(x2,y2) = np.floor(detection['box'][0:2]).astype(int), np.ceil(detection['box'][2:4]).astype(int)
		color = (255, 0, 0)


		#label = str(int(detection['x_world'])) + '; ' + str(int(detection['y_world']))
		#t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
		#cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
		#cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
		
		# draw BBox
		cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
	return img

sender = DetectionImageSender(connect_to='tcp://localhost:5556')

frame1 = cv2.imread("/home/dobreff/videos/samples/1027_14_25_53_first.bmp")
frame2 = cv2.imread("/home/dobreff/videos/samples/1026_05_46_59_first.bmp")
frame3 = cv2.imread("/home/dobreff/videos/samples/1022_07_51_42_first.bmp")

frames = [frame1, frame2, frame3]

for idx, img in enumerate(frames):
	st = time.time()
	list_result = sender.send_image('blabla', img)
	recv = time.time()
	print(recv - st)
	# newImg = draw_bboxes(img, list_result)
	# cv2.imwrite(f'/home/dobreff/videos/output2_{idx}.jpg', newImg)



