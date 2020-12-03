from threading import Thread, Event
from queue import Queue
import requests

class DetectronClient():
	class ClientState():
		Stopped = 0
		Started = 1

	def __init__(self, address, queue, name, delay):
		self.clientThread = Thread(target=self.senderFunc, args=(), daemon=True)
		self.address = address
		self.state = self.ClientState.Stopped
		self.name = name
		self.Q = queue
		# DEBUG
		self.delay = delay
		self.address = f'http://slowwly.robertomurray.co.uk/delay/{delay}/url/https://vanenet.hu/'

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
			if not self.Q.empty():
				item = self.Q.get()
				resp = requests.get(self.address)
				print(f'Item {item} processed by {self.name}')
			else:
				print('Queue empty, waiting...', self.name)
				time.sleep(0.5)
				
			if self.state == self.ClientState.Stopped:
				break
				
	def stop(self):
		# indicate that the thread should be stopped
		self.state = self.ClientState.Stopped
		# wait until stream resources are released (producer thread might be still grabbing frame)
		self.clientThread.join()