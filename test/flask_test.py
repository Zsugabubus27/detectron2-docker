import time
import numpy as np
from detectron_client import DetectronClient
import cv2
from utils import draw_bboxes
from queue import Queue
import pickle


# Fill up Queue of images
_tmpList = ["/home/dobreff/videos/samples/1027_14_25_53_first.bmp", 
			"/home/dobreff/videos/samples/1026_05_46_59_first.bmp", 
			"/home/dobreff/videos/samples/1022_07_51_42_first.bmp"] * 1000

NUM_OF_FRAMES = len(_tmpList)

queue_frames = Queue()

for idx, imgPath in enumerate(_tmpList):
	frame = cv2.imread(imgPath)
	queue_frames.put({'idx' : idx, 'image' : frame})

# Init output queue
queue_results = Queue()

# Initialize Detectron Clients
dc1 = DetectronClient('localhost:5555', queue_frames, queue_results)
dc2 = DetectronClient('localhost:5556', queue_frames, queue_results)
dc3 = DetectronClient('localhost:5557', queue_frames, queue_results)
dc4 = DetectronClient('localhost:5558', queue_frames, queue_results)

st = time.time()
# Start processing
dc1.start()
dc2.start()
dc3.start()
dc4.start()



# Wait to process every frame:
while queue_results.qsize() < NUM_OF_FRAMES:
	time.sleep(1)

print(time.time() - st, queue_results.qsize())

# Save results
dict_results = {}
while (not queue_results.empty()):
	idx, img_orig, list_res = queue_results.get()
	dict_results[idx] = list_res
	newImg = draw_bboxes(img_orig, list_res)
	cv2.imwrite(f'/home/dobreff/videos/outputs/test/{idx}.jpg', newImg)
	print(f'Item {idx} saved!')

with open('/home/dobreff/videos/outputs/test/test.pickle', 'wb') as handle:
	pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Mivel maxra fel van töltve a Queue, addig pörgök a ciklusban amíg van benne elem:
# while (not queue_frames.empty()) or (not queue_results.empty()):
# 	# Grab result image
# 	idx, img_orig, list_res = queue_results.get()
# 	newImg = draw_bboxes(img_orig, list_res)
# 	cv2.imwrite(f'/home/dobreff/videos/outputs/{idx}.jpg', newImg)
# 	print(f'Item {idx} saved!')


# for idx, img in enumerate(frames):
# 	st = time.time()
# 	list_result = sender.send_image('blabla', img)
# 	recv = time.time()
# 	print(recv - st)
# 	newImg = draw_bboxes(img, list_result)
# 	cv2.imwrite(f'/home/dobreff/videos/output2_{idx}.jpg', newImg)



