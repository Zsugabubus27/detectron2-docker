import numpy as np
import cv2
import time
import torch, torchvision
import coord_mapper
# Some basic setup:

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances, pairwise_iou

print('Torch version:', torch.__version__, torch.cuda.is_available())


class PlayerDetectorDetectron2():
	def __init__(self, bg_history, leftSideCorners, rightSideCorners, 
				coordMapper,
				origResolution, outResolution, segnumx, segnumy, nmsThreshold):
		
		
		# Coordinate Mapper class
		self.coordMapper = coordMapper
		
		self.nmsThreshold = nmsThreshold

		# Infos about the frames
		# Width is the double the original since we hconcated the frames
		self.origResolution = (origResolution[0] * 2, origResolution[1]) # Width, Height
		self.outResolution = (outResolution[0] * 2, outResolution[1]) # Width, Height
		self.trans_value = float(self.outResolution[0]) / self.origResolution[0]

		# TODO: Create model
		model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
		self.cfg = get_cfg()
		# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
		self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
		# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
		self.predictor = DefaultPredictor(self.cfg)

		# TODO: Tesztelésképp csak y irányba kettévágom 2x x irányban is, 
		# és egy átlapolódó rácsot is készitek 
		# Grid = (xTL, yTL, xBR, yBR)
		self.gridList = [(0, 0, 5120//2, 1440 // 2), (5120//2, 0, 5120, 1440 // 2),
						(0, 1440 // 2, 5120//2, 1440), (5120//2, 1440 // 2, 5120, 1440),
						(0, 1440 // 3, 5120//2, 2*1440 // 3), (5120//2, 1440 // 3, 5120, 2*1440 // 2)]

		# im1 = origImg[:1440 // 2, 5120//2 : 3 * 5120//4]
		# im2 = origImg[:1440 // 2, 5120//4 : 2 * 5120//4]
		# im3 = origImg[1440 // 2:, 5120//2 : 3 * 5120//4]
		# im4 = origImg[1440 // 2:, 5120//4 : 2 * 5120//4]
		# cv2_imshow(im4)

		# # hány background modellt készítsünk x és y irányban
		# self.segnumx = segnumx
		# self.segnumy = segnumy

		# # A rács celláinak oldalhosszai 
		# self.segmentation_x = self.outResolution[0] // self.segnumx
		# self.segmentation_y = self.outResolution[1] // self.segnumy

		# # Init Background subtractor list
		# self.BGSubstractorList = {}
		# # Bal felső sarokból indulva indítom a rácsot
		# for x_cut in range(0, self.outResolution[0], self.segmentation_x):
		# 	for y_cut in range(0, self.outResolution[1], self.segmentation_y):
		# 		val_x = self.segmentation_x if x_cut + self.segmentation_x <= self.outResolution[0] else self.outResolution[0] - x_cut
		# 		val_y = self.segmentation_y if y_cut + self.segmentation_y <= self.outResolution[1] else self.outResolution[1] - y_cut
		# 		self.BGSubstractorList[(x_cut, y_cut)] = (cv2.createBackgroundSubtractorMOG2(history=bg_history), val_x, val_y)
		# if not(self.segnumx == 2 and self.segnumy == 1):
		# 	# X irányba eltolva indítom a rácsot
		# 	for x_cut in range(int(self.segmentation_x / 2), self.outResolution[0], self.segmentation_x):
		# 		for y_cut in range(0, self.outResolution[1], self.segmentation_y):
		# 			val_x = self.segmentation_x if x_cut + self.segmentation_x <= self.outResolution[0] else self.outResolution[0] - x_cut
		# 			val_y = self.segmentation_y if y_cut + self.segmentation_y <= self.outResolution[1] else self.outResolution[1] - y_cut
		# 			self.BGSubstractorList[(x_cut, y_cut)] = (cv2.createBackgroundSubtractorMOG2(history=bg_history), val_x, val_y)

		# 	# Y irányba eltolva indítom a rácsot
		# 	for x_cut in range(0, self.outResolution[0], self.segmentation_x):
		# 		for y_cut in range(int(self.segmentation_y / 2), self.outResolution[1], self.segmentation_y):
		# 			val_x = self.segmentation_x if x_cut + self.segmentation_x <= self.outResolution[0] else self.outResolution[0] - x_cut
		# 			val_y = self.segmentation_y if y_cut + self.segmentation_y <= self.outResolution[1] else self.outResolution[1] - y_cut
		# 			self.BGSubstractorList[(x_cut, y_cut)] = (cv2.createBackgroundSubtractorMOG2(history=bg_history), val_x, val_y)

		self.fieldPolygon = self._getFieldBoundary(leftSideCorners, rightSideCorners)
	
	def _getFieldBoundary(self, left_side_corner_pixels, right_side_corner_pixels_added_half_field):
		# Azok azért vannak, hogy aki a pálya szélén fut a szélső oldalvonalnál, kell egy kis overhead, mert ha a lába van a vonalnál, a feje már nem fér bele, sőt, a dereka sem ~ Marci)
		merged_arr = [[left_side_corner_pixels[0][0],left_side_corner_pixels[0][1]-25],
				[left_side_corner_pixels[1][0],left_side_corner_pixels[1][1]-25],
				[right_side_corner_pixels_added_half_field[0][0],right_side_corner_pixels_added_half_field[0][1]-25],
				[right_side_corner_pixels_added_half_field[1][0],right_side_corner_pixels_added_half_field[1][1]-25],
				[right_side_corner_pixels_added_half_field[2][0],right_side_corner_pixels_added_half_field[2][1]],
				right_side_corner_pixels_added_half_field[3],
				left_side_corner_pixels[2],
				left_side_corner_pixels[3]]
		
		merged_arr = np.array(merged_arr) * (float(self.outResolution[0]) / self.origResolution[0]) # outResolution

		return np.array(merged_arr, dtype=np.int32)

	def _getTeamColor(self, full_frame, bbox):
		x,y,w,h = bbox
		img = full_frame[y:y+h, x:x+w]

		## convert to hsv
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
		#Give the official range of the two teams shirt colors
		#yellow
		yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))

		#red
		# we have two red, because the red value is the beginning = final value of the hsv cylinder
		red1 = cv2.inRange(hsv, (0, 100, 100),(10, 255, 255))
		red2 = cv2.inRange(hsv, (160, 100, 100),(179, 255, 255))
		red = red1 | red2

		h,w,_ = hsv.shape
		sum_masked = h*w
		yellow_percentage = cv2.countNonZero(yellow)*100/sum_masked
		red_percentage =  cv2.countNonZero(red)*100/sum_masked

	
		if yellow_percentage > 5 and red_percentage > 5:
			return "more player from different team"
		elif yellow_percentage > 5 or red_percentage > 5:
			return  "yellow" if  yellow_percentage > red_percentage else "red"
		else:
			return "other"

	def preprocess(self, image):
		'''
		Deletes the outer region of the field
		'''
		# Create mask for the field
		null_frame = np.zeros([image.shape[0], image.shape[1]],dtype=np.uint8)
		cv2.fillPoly(null_frame, [self.fieldPolygon], 255 )
		# Masks out the outer region of the field
		frame = cv2.bitwise_and(image, image, mask=null_frame)
		return frame

	def _predictMultipleImages(self, images : list):
		"""
		Args:
			list of original_images List[np.ndarray]: an image of shape (H, W, C) (in BGR order).
		Returns:
			predictions (list):
				list of predictions for every image of images list
		"""
		st = time.time()
		# Disable gradient calculation
		with torch.no_grad():
			# Apply pre-processing to every image
			if self.predictor.input_format == "RGB":
				# whether the model expects BGR inputs or RGB
				images = [img[:, :, ::-1] for img in images]
			# create inputs for the Model
			inputs = [{"image": img, "height": img.shape[0], "width": img.shape[1]} for img in images]
			for imgDict in inputs:
				imgDict['image'] = self.predictor.aug.get_transform(imgDict['image']).apply_image(imgDict['image'])
				imgDict['image'] = torch.as_tensor(imgDict['image'].astype("float32").transpose(2, 0, 1))

			predictions = self.predictor.model(inputs)
			print('pred:', time.time() - st)
			return predictions

	def _cutImageToGrids(self, image):
		# Grid = (xTL, yTL, xBR, yBR)
		return [image[yTL : yBR, xTL : xBR] for xTL, yTL, xBR, yBR in self.gridList]

		

	def getForegroundContours(self, image):
		'''
		image: Frame amin a játékosokat akarjuk megtalálni
		result: dict((x_cut, y_cut) -> counturList)

		'''
		# 0. Preprocess image
		frame = self.preprocess(image)
		
		# 1. Külön felvágom a képeket kis cellákra
		l_cells = self._cutImageToGrids(frame)
		
		# 2. Majd ezeket a képeket beadom a multiple prediktálóba
		l_preds = self._predictMultipleImages(l_cells)
		
		# 2.1 Lista a cellákon található instancokról
		l_preds = [x['instances'].to('cpu') for x in l_preds]
		
		# 2.2 Visszamappelem az eredeti képre & pred_masks nincs használva
		# 2.3 A Kamerától vett távolság függvényében változtatom a score-t (yTL)
		cameraDistWeight = reversed(np.unique(np.asarray(self.gridList)[:, 1]))
		cameraDistWeight = {yCord : (100-idx) / 100 for idx, yCord in enumerate(cameraDistWeight)}
		for inst, cell in zip(l_preds, self.gridList): 
			inst.remove('pred_masks')
			inst.pred_boxes.tensor[:, 0:4] += torch.Tensor([cell[0], cell[1], cell[0], cell[1]])
			inst.scores *= cameraDistWeight[cell[1]]
		
		# 2.4 Egész képre vonatkoztatott Instancok
		# TODO: Hiányzik a resize!!!!!!
		# TODO: Most akkor itt origResolution??? scalelés hol történjen???
		finalInstances = Instances(image_size=self.origResolution[::-1]) # (1440, 5120)
		finalInstances.pred_boxes = Boxes.cat([x.pred_boxes for x in l_preds])
		finalInstances.scores = torch.cat([x.scores for x in l_preds])
		finalInstances.pred_classes = torch.cat([x.pred_classes for x in l_preds])

		# 3. Leszűröm az emberekre csak
		_person_class_ID = 0
		finalInstances.pred_boxes.tensor = finalInstances.pred_boxes.tensor[finalInstances.pred_classes == _person_class_ID]
		finalInstances.scores = finalInstances.scores[finalInstances.pred_classes == _person_class_ID]
		finalInstances.pred_classes = finalInstances.pred_classes[finalInstances.pred_classes == _person_class_ID]
		
		# 4. NMS használata, hogy kiiktassam az átlapolódásokat
		iouIdx = torchvision.ops.nms(finalInstances.pred_boxes.tensor, finalInstances.scores, self.nmsThreshold)
		finalInstances.pred_boxes.tensor = finalInstances.pred_boxes.tensor[iouIdx]
		finalInstances.scores = finalInstances.scores[iouIdx]
		finalInstances.pred_classes = finalInstances.pred_classes[iouIdx]
		
		
		return finalInstances
	
	def detectPlayersOnFrame(self, frame):

		# 1. Detektálom a framen a játékosokat
		allInstances = self.getForegroundContours(frame)
		
		# 2. Kiszámolom a valós koordinátájukat
		worldcoords_xy = self.coordMapper.image2xy([( (box[0] + box[2]) / 2, box[3]) for box in allInstances.pred_boxes.tensor])

		v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
		v.draw_instance_predictions(allInstances)
		for worldXY, box in zip(worldcoords_xy, allInstances.pred_boxes.tensor):
			strToDraw = "{0:.1f}; {1:.1f}".format(*worldXY) if worldXY is not None else 'XXX'
			v.draw_text(strToDraw, ( (box[0] + box[2]) / 2, (box[1] + box[3]) / 2), font_size=11)
		cv2.imwrite('/tmp/outputs/merged_wWorldCoord.jpg', v.get_output().get_image()[:, :, ::-1])

		# v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
		# out = v.draw_instance_predictions(allInstances)
		# cv2.imwrite('/tmp/outputs/merged.jpg', out.get_image()[:, :, ::-1])


		# for idx, (img, pred) in enumerate(l_results):
		# 	v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
		# 	out = v.draw_instance_predictions(pred["instances"].to("cpu"))
		# 	cv2.imwrite(f'/tmp/outputs/{idx}.jpg', out.get_image()[:, :, ::-1])

		# l_results = self.getForegroundContours(frame)
		# for idx, (img, pred) in enumerate(l_results):
		# 	v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
		# 	out = v.draw_instance_predictions(pred["instances"].to("cpu"))
		# 	cv2.imwrite(f'/tmp/outputs/{idx}.jpg', out.get_image()[:, :, ::-1])
			#cv2_imshow(out.get_image()[:, :, ::-1])

		# # (x_cut, y_cut) -> contours
		# contoursDict = self.getForegroundContours(frame)

		# detectedPlayers = []
		# # LOOP ONE EACH CELL's each CONTOUR
		# for (x_cut, y_cut), contours in contoursDict.items():
			
		# 	# LOOP Through every contour of the given cell
		# 	for cnt in contours:
				
		# 		# x, y, w, h: A resizelt kép kis cellájára vonatkoznak ezek az értékek
		# 		x_cell, y_cell, w_cell, h_cell = cv2.boundingRect(cnt)

		# 		# Visszamappelem az eredeti képre (cellák és resize leküzdése)
		# 		x, y = (x_cell + x_cut) // self.trans_value, (y_cell + y_cut) //  self.trans_value 
		# 		w, h = (w_cell // self.trans_value), (h_cell // self.trans_value)

		# 		# Validate the minimum dimensional requirements
		# 		if w > self.min_width_of_human_in_pixel and h > self.min_height_of_human_in_pixel and h > w:
		# 			centerPoint = [x+(w/2), y+(h/2)] # Center point az eredeti képen
		# 			max_w, min_w = self.validatorFunction(centerPoint)
					
		# 			if w > min_w and w < max_w:
		# 				# Átmappelem csak a nagy képre, DE nem skálázom vissza az eredeti képre
		# 				team = self._getTeamColor(frame, (x_cell + x_cut, y_cell + y_cut, w_cell, h_cell))

		# 				detectedPlayers.append( {'x':int(x), 'y': int(y), 'w' : int(w), 'h' : int(h), 'color' : team} )
		
		# # Leszűröm az átlapolódó játékosokat (top left x,y,w,h) a bemenete
		# _, mask_nms = nms.nms([(p['x'], p['y'], p['w'], p['h']) for p in detectedPlayers ], threshold=0.7)
		# detectedPlayers = [p for p, b in zip(detectedPlayers, mask_nms) if b]
		
		# # Kiszámolom az összes detektált játékosra a világkoordinátában lévő pontjaikat
		# worldcoords_xy = self.coordMapper.image2xy([(p['x'] + (p['w'] / 2), p['y'] + p['h']) for p in detectedPlayers])
		# for p, wcoords in zip(detectedPlayers, worldcoords_xy):
		# 	p.update({'x_world' : wcoords[0], 'y_world' : wcoords[1]})

		# return detectedPlayers

left_side_corner_pixels =  [[1264.7023,499.5309],[1870.1519,549.1178 ],[2517.0312,1275.4386 ],[ 192.16057,548.8432 ]]
right_side_corner_pixels = [[755.81616,474.7149],[1361.8402,456.584],[2482.6863,568.66864],[257.55048,1084.8362]]
full_width = (2560, 1440)[0]*2
right_side_corner_pixels_added_half_field = np.array([[p[0] + full_width/2, p[1]] for p in right_side_corner_pixels])

# CoordMapper a pálya széleivel inicializálva
coordMapper = coord_mapper.CoordMapperCSG(match_code=(left_side_corner_pixels, right_side_corner_pixels))


myDetector = PlayerDetectorDetectron2(0, left_side_corner_pixels, right_side_corner_pixels_added_half_field, 
								coordMapper, (2560, 1440), (2560, 1440), 0, 0, 0.5)

myFrame = cv2.imread("/tmp/sample_5k.bmp")

myDetector.detectPlayersOnFrame(myFrame)
print('done')