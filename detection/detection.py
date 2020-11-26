import numpy as np
import cv2
import time
import nms

class PlayerDetectorDetectron2():
	def __init__(self, bg_history, validatorFunction, leftSideCorners, rightSideCorners, 
				min_width_of_human_in_pixel, min_height_of_human_in_pixel, coordMapper,
				origResolution, outResolution, segnumx, segnumy):
		
		# Dynamic Validation function
		self.validatorFunction = validatorFunction
		
		# Coordinate Mapper class
		self.coordMapper = coordMapper

		self.min_width_of_human_in_pixel = min_width_of_human_in_pixel
		self.min_height_of_human_in_pixel = min_height_of_human_in_pixel
		
		# Infos about the frames
		# Width is the double the original since we hconcated the frames
		self.origResolution = (origResolution[0] * 2, origResolution[1]) # Width, Height
		self.outResolution = (outResolution[0] * 2, outResolution[1]) # Width, Height
		self.trans_value = float(self.outResolution[0]) / self.origResolution[0]

		# hány background modellt készítsünk x és y irányban
		self.segnumx = segnumx
		self.segnumy = segnumy

		# A rács celláinak oldalhosszai 
		self.segmentation_x = self.outResolution[0] // self.segnumx
		self.segmentation_y = self.outResolution[1] // self.segnumy

		# Init Background subtractor list
		self.BGSubstractorList = {}
		# Bal felső sarokból indulva indítom a rácsot
		for x_cut in range(0, self.outResolution[0], self.segmentation_x):
			for y_cut in range(0, self.outResolution[1], self.segmentation_y):
				val_x = self.segmentation_x if x_cut + self.segmentation_x <= self.outResolution[0] else self.outResolution[0] - x_cut
				val_y = self.segmentation_y if y_cut + self.segmentation_y <= self.outResolution[1] else self.outResolution[1] - y_cut
				self.BGSubstractorList[(x_cut, y_cut)] = (cv2.createBackgroundSubtractorMOG2(history=bg_history), val_x, val_y)
		if not(self.segnumx == 2 and self.segnumy == 1):
			# X irányba eltolva indítom a rácsot
			for x_cut in range(int(self.segmentation_x / 2), self.outResolution[0], self.segmentation_x):
				for y_cut in range(0, self.outResolution[1], self.segmentation_y):
					val_x = self.segmentation_x if x_cut + self.segmentation_x <= self.outResolution[0] else self.outResolution[0] - x_cut
					val_y = self.segmentation_y if y_cut + self.segmentation_y <= self.outResolution[1] else self.outResolution[1] - y_cut
					self.BGSubstractorList[(x_cut, y_cut)] = (cv2.createBackgroundSubtractorMOG2(history=bg_history), val_x, val_y)

			# Y irányba eltolva indítom a rácsot
			for x_cut in range(0, self.outResolution[0], self.segmentation_x):
				for y_cut in range(int(self.segmentation_y / 2), self.outResolution[1], self.segmentation_y):
					val_x = self.segmentation_x if x_cut + self.segmentation_x <= self.outResolution[0] else self.outResolution[0] - x_cut
					val_y = self.segmentation_y if y_cut + self.segmentation_y <= self.outResolution[1] else self.outResolution[1] - y_cut
					self.BGSubstractorList[(x_cut, y_cut)] = (cv2.createBackgroundSubtractorMOG2(history=bg_history), val_x, val_y)

		# Initializing the two kernels used for morphological operations
		# OR cv2.MORPH_ELLIPSE,(3,1)
		self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,1))
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

	
	def getForegroundContours(self, image):
		frame = self.preprocess(image)
		
		resultDict = {}
		# width = self.outResolution[0], height = self.outResolution[1]
		for (x_cut, y_cut), (bgModel, cell_width, cell_height) in self.BGSubstractorList.items():
			#Kivágom a képet az eredeti nagy képről.
			cutted_frame = frame[y_cut : y_cut + cell_height, x_cut : x_cut + cell_width] 
			#GET MASK FROM CUTTED FRAME
			mask = bgModel.apply(cutted_frame)
			
			#NOISE REDUCTION ON MASK
			fgmask_matrix = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

			# REMOVE SHADOWS
			_, fgmask_matrix = cv2.threshold(fgmask_matrix, 127, 255, cv2.THRESH_BINARY)

			# Get every contour on cutted_frame
			# contours: List of (x,y) coords of the contour
			contours, _ = cv2.findContours(fgmask_matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			resultDict[(x_cut, y_cut)] = contours
		
		return resultDict
	
	def detectPlayersOnFrame(self, frame):
		
		# (x_cut, y_cut) -> contours
		contoursDict = self.getForegroundContours(frame)

		detectedPlayers = []
		# LOOP ONE EACH CELL's each CONTOUR
		for (x_cut, y_cut), contours in contoursDict.items():
			
			# LOOP Through every contour of the given cell
			for cnt in contours:
				
				# x, y, w, h: A resizelt kép kis cellájára vonatkoznak ezek az értékek
				x_cell, y_cell, w_cell, h_cell = cv2.boundingRect(cnt)

				# Visszamappelem az eredeti képre (cellák és resize leküzdése)
				x, y = (x_cell + x_cut) // self.trans_value, (y_cell + y_cut) //  self.trans_value 
				w, h = (w_cell // self.trans_value), (h_cell // self.trans_value)

				# Validate the minimum dimensional requirements
				if w > self.min_width_of_human_in_pixel and h > self.min_height_of_human_in_pixel and h > w:
					centerPoint = [x+(w/2), y+(h/2)] # Center point az eredeti képen
					max_w, min_w = self.validatorFunction(centerPoint)
					
					if w > min_w and w < max_w:
						# Átmappelem csak a nagy képre, DE nem skálázom vissza az eredeti képre
						team = self._getTeamColor(frame, (x_cell + x_cut, y_cell + y_cut, w_cell, h_cell))

						detectedPlayers.append( {'x':int(x), 'y': int(y), 'w' : int(w), 'h' : int(h), 'color' : team} )
		
		# Leszűröm az átlapolódó játékosokat (top left x,y,w,h) a bemenete
		_, mask_nms = nms.nms([(p['x'], p['y'], p['w'], p['h']) for p in detectedPlayers ], threshold=0.7)
		detectedPlayers = [p for p, b in zip(detectedPlayers, mask_nms) if b]
		
		# Kiszámolom az összes detektált játékosra a világkoordinátában lévő pontjaikat
		worldcoords_xy = self.coordMapper.image2xy([(p['x'] + (p['w'] / 2), p['y'] + p['h']) for p in detectedPlayers])
		for p, wcoords in zip(detectedPlayers, worldcoords_xy):
			p.update({'x_world' : wcoords[0], 'y_world' : wcoords[1]})

		return detectedPlayers
   
