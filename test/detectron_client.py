from threading import Thread, Event
from utils import DetectionImageSender
import time

class DetectronClient():
	class ClientState():
		Stopped = 0
		Started = 1

	def __init__(self, address, input_queue, output_queue, verbose=True, storeImg = False):
		self.clientThread = Thread(target=self.senderFunc, args=(), daemon=True)
		self.address = address
		self.sender = DetectionImageSender(connect_to=f'tcp://{self.address}')
		self.state = self.ClientState.Stopped
		self.Q_in = input_queue
		self.Q_out = output_queue
		self.verbose = verbose
		self.storeImg = storeImg

	def start(self):
		if self.state != self.ClientState.Stopped:
			raise ValueError
		self.state = self.ClientState.Started
		self.clientThread.start()
		return self

	def senderFunc(self):
		while True:
			# Folyton kiszedek a Sorból egy képet és elküldöm a Detectron konténernek
			# Ensure the queue is not empty
			if not self.Q_in.empty():
				item = self.Q_in.get()
				sendMsg = str(item['resolution']) + '_' + str(item['idx'])
				list_result = self.sender.send_image(sendMsg, item['image'])
				if self.storeImg:
					self.Q_out.put((item['idx'], item['image'], list_result))
				else:
					self.Q_out.put((item['idx'], list_result))
				
				if self.verbose: print(f'Item {item["idx"]} processed by {self.address}')
			else:
				if self.verbose: print('Input queue empty, waiting...', self.address)
				time.sleep(1)
				
			if self.state == self.ClientState.Stopped:
				break
				
	def stop(self):
		# indicate that the thread should be stopped
		self.state = self.ClientState.Stopped
		# wait until stream resources are released (producer thread might be still grabbing frame)
		self.clientThread.join()