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
	def __init__(self, leftSideCorners, rightSideCorners, coordMapper,
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

		# Hány cella legyen x (width) és y (height) irányba
		# x irányba bal és jobb oldalt is segnumx darab cella lesz
		self.segnumx = segnumx
		self.segnumy = segnumy

		# A rács celláinak oldalhosszai 
		self.cell_width = (self.outResolution[0] // 2) // self.segnumx
		self.cell_height = self.outResolution[1] // self.segnumy
		
		# Rács elkészítése
		# Grid = (xTL, yTL, xBR, yBR)
		#self.gridList = self._createGridCells()
		#self.gridList = [[222, 454, 2153, 787], [465, 633, 4825, 1439], [3117, 427, 5119, 800]]
		#self.gridList = [[220, 453, 1466, 666], [1408, 433, 3739, 750], [3691, 431, 5119, 738], [421, 600, 4569, 1439]]
		self.gridList = [[221, 470, 2029, 655], [356, 575, 2560, 1439], [2560, 502, 5117, 1330], [3224, 434, 4613, 559]]
		print('grids', len(self.gridList))
		
		#A Kamerától vett távolság függvényében változtatom a score-t (yTL)
		self.cameraDistWeight = reversed(np.unique(np.asarray(self.gridList)[:, 1]))
		self.cameraDistWeight = {yCord : (100-idx) / 100 for idx, yCord in enumerate(self.cameraDistWeight)}


		self.fieldPolygon = self._getFieldBoundary(leftSideCorners, rightSideCorners)
	
	def _getFieldBoundary(self, left_side_corner_pixels, right_side_corner_pixels_added_half_field):
		# Azok azért vannak, hogy aki a pálya szélén fut a szélső oldalvonalnál, kell egy kis overhead, mert ha a lába van a vonalnál, a feje már nem fér bele, sőt, a dereka sem ~ Marci)
		merged_arr = [[223,457],[1261,474],[1914,522],[2560,863],[3305,437],[3937,435],[5119,468],
					[5119,546],[2560,1328],[2560,1439],[2486,1439],[223,553]]

		# merged_arr = [[left_side_corner_pixels[0][0],left_side_corner_pixels[0][1]-25],
		# 		[left_side_corner_pixels[1][0],left_side_corner_pixels[1][1]-25],
		# 		[right_side_corner_pixels_added_half_field[0][0],right_side_corner_pixels_added_half_field[0][1]-25],
		# 		[right_side_corner_pixels_added_half_field[1][0],right_side_corner_pixels_added_half_field[1][1]-25],
		# 		[right_side_corner_pixels_added_half_field[2][0],right_side_corner_pixels_added_half_field[2][1]],
		# 		right_side_corner_pixels_added_half_field[3],
		# 		left_side_corner_pixels[2],
		# 		left_side_corner_pixels[3]]
		
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
			return predictions

	def _cutImageToGrids(self, image):
		# Grid = (xTL, yTL, xBR, yBR)
		return [image[yTL : yBR, xTL : xBR] for xTL, yTL, xBR, yBR in self.gridList]

	def _createGridCells(self):
		# Grid = (xTL, yTL, xBR, yBR)
		gridList = []
		sideWidth = self.outResolution[0] // 2
		# 1. Alaprács elkészítése
		for xTL in range(0, self.outResolution[0], self.cell_width):
			for yTL in range(0, self.outResolution[1], self.cell_height):
				gridList.append((xTL, yTL, xTL + self.cell_width, yTL + self.cell_height))
		# 2. Függőleges oldalak mentén bal és jobb oldalra is
		for col in range(0, self.segnumx - 1):
			for row in range(0, self.segnumy):
				xTL = self.cell_width // 2 + col * self.cell_width
				yTL = row * self.cell_height
				gridList.append( (xTL, yTL, xTL + self.cell_width, yTL + self.cell_height) )
				gridList.append( (xTL + sideWidth, yTL, 
								xTL + self.cell_width + sideWidth, yTL + self.cell_height) )
		
		# 3. Vízszintes oldalak mentén bal és jobb oldalra is
		for col in range(0, self.segnumx):
			for row in range(0, self.segnumy - 1):
				xTL = col * self.cell_width
				yTL = (self.cell_height // 2) + row * self.cell_height
				gridList.append( (xTL, yTL, xTL + self.cell_width, yTL + self.cell_height) )
				gridList.append( (xTL + sideWidth, yTL, 
								xTL + self.cell_width + sideWidth, yTL + self.cell_height) )
		# 4. Metszéspontok mentén bal és jobb oldalra is
		for col in range(0, self.segnumx - 1):
			for row in range(0, self.segnumy - 1):
				xTL = (self.cell_width // 2) + col * self.cell_width
				yTL = (self.cell_height // 2) + row * self.cell_height
				gridList.append( (xTL, yTL, xTL + self.cell_width, yTL + self.cell_height) )
				gridList.append( (xTL + sideWidth, yTL, 
								xTL + self.cell_width + sideWidth, yTL + self.cell_height) )

		return gridList

	def _detectAndMap(self, image):
		'''
		image: Frame amin a játékosokat akarjuk megtalálni
		result: dict((x_cut, y_cut) -> counturList)

		'''
		st = time.time()
		# 0. Preprocess image
		frame = self.preprocess(image)

		# 1. Külön felvágom a képeket kis cellákra
		l_cells = self._cutImageToGrids(frame)


		# TODO: DEBUG
		# for idx, img in enumerate(l_cells):
		# 	cv2.imwrite(f'/tmp/outputs/c_{idx}.jpg', img)

		# 2. Majd ezeket a képeket beadom a multiple prediktálóba
		l_preds = self._predictMultipleImages(l_cells)

		# 2.1 Lista a cellákon található instancokról
		l_preds = [x['instances'].to('cpu') for x in l_preds]

		# 2.2 Visszamappelem a bemeneti képre, majd felskálázom az 5K-s képre
		# 2.3 A Kamerától vett távolság függvényében változtatom a score-t (yTL)
		for inst, cell in zip(l_preds, self.gridList): 
			inst.remove('pred_masks') # pred_masks nincs használva TODO: TeamColor esetében lehet jól jön
			inst.pred_boxes.tensor[:, 0:4] += torch.Tensor([cell[0], cell[1], cell[0], cell[1]])
			inst.boxes_before = inst.pred_boxes.clone() # Eredeti képen skálázás nélkül hol vannak
			inst.pred_boxes.tensor = inst.pred_boxes.tensor.divide(self.trans_value)
			inst.scores *= self.cameraDistWeight[cell[1]]


		# 2.4 Egész képre vonatkoztatott Instancok
		finalInstances = Instances(image_size=self.origResolution[::-1]) # (1440, 5120)
		finalInstances.pred_boxes = Boxes.cat([x.pred_boxes for x in l_preds])
		finalInstances.boxes_before = Boxes.cat([x.boxes_before for x in l_preds])
		finalInstances.scores = torch.cat([x.scores for x in l_preds])
		finalInstances.pred_classes = torch.cat([x.pred_classes for x in l_preds])


		# 3. Leszűröm az emberekre csak
		_person_class_ID = 0
		finalInstances = finalInstances[finalInstances.pred_classes == _person_class_ID]

		# 4. NMS használata, hogy kiiktassam az átlapolódásokat
		iouIdx = torchvision.ops.nms(finalInstances.pred_boxes.tensor, finalInstances.scores, self.nmsThreshold)
		finalInstances = finalInstances[iouIdx]

		return finalInstances, frame
	
	def detectPlayersOnFrame(self, frame):

		# 1. Detektálom a framen a játékosokat
		allInstances, frame = self._detectAndMap(frame)

		# Ide már a pred_boxes-ban lévő koordinátáknak a 1440x5120-es dimenzióban kell lenni, mert úgy van implementálva a class
		# 2. Kiszámolom a valós koordinátájukat
		worldcoords_xy = self.coordMapper.image2xy([( (box[0] + box[2]) / 2, box[3]) for box in allInstances.pred_boxes.tensor])

		# 3. Leszűröm a középen lévő játékosokat
		maskWorldCoord = [ True if x is not None else False for x in worldcoords_xy]
		allInstances = allInstances[maskWorldCoord]
		worldcoords_xy = [x for x in worldcoords_xy if x is not None]

		# 4. Detekciókat tartalmazó lista létrehozása
		list_result = []
		for bigBox, smallBox, score, worldXY in zip(allInstances.pred_boxes.tensor.numpy(), 
												allInstances.boxes_before.tensor.numpy(),
												allInstances.scores.numpy(), 
												worldcoords_xy):
			(xTL, yTL), (xBR, yBR)  = np.floor(smallBox[0:2]).astype(int), np.ceil(smallBox[2:4]).astype(int)
			clipped_img = frame[yTL : yBR, xTL : xBR]
			list_result.append( {'worldXY' : worldXY, 'box' : smallBox, 
								'bigBox' : bigBox, 'score' : score, 'image' : clipped_img} )
		return list_result


if __name__ == '__main__':
	origRes = (2560, 1440)
	outRes = (1920, 1080) #  ["(2560, 1440)", "(2048, 1152)", "(1920, 1080)", "(1536, 864)", "(1280, 720)"] 
	
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

	myFrame = cv2.imread("/tmp/sample_5k.bmp")
	# Ez már a konkatenált kép, ezért 2x-es a szélessége
	myFrame = cv2.resize(myFrame, (outRes[0]*2, outRes[1]), interpolation = cv2.INTER_AREA)
	myDetector.detectPlayersOnFrame(myFrame)
	print('done')
	myDetector.detectPlayersOnFrame(myFrame)
	print('done')
	myDetector.detectPlayersOnFrame(myFrame)
	print('done')
	myDetector.detectPlayersOnFrame(myFrame)
	print('done')