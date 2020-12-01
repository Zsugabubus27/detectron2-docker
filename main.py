from flask import Flask, request, Response
import cv2
import numpy as np
import coord_mapper
from detection import PlayerDetectorDetectron2
# Define a flask app
app = Flask(__name__)
myDetector = None # TODO:

# route http posts to this method
@app.route('/detect', methods=['POST'])
def infer():
	print('A')
	myFrame = cv2.imread("/tmp/sample_5k.bmp")
	print('B')
	myDetector.detectPlayersOnFrame(myFrame)
	print('C')
	return 'lol'

# route http posts to this method
@app.route('/', methods=['GET'])
def hello():
	return "Halika tesssaa"

def initApplication():
	origRes = (2560, 1440)
	outRes = (2560, 1440) #  ["(2560, 1440)", "(2048, 1152)", "(1920, 1080)", "(1536, 864)", "(1280, 720)"] 
	
	left_side_corner_pixels =  [[1264.7023,499.5309],[1870.1519,549.1178 ],[2517.0312,1275.4386 ],[ 192.16057,548.8432 ]]
	right_side_corner_pixels = [[755.81616,474.7149],[1361.8402,456.584],[2482.6863,568.66864],[257.55048,1084.8362]]
	full_width = origRes[0]*2
	right_side_corner_pixels_added_half_field = np.array([[p[0] + full_width/2, p[1]] for p in right_side_corner_pixels])

	# CoordMapper a pálya széleivel inicializálva
	coordMapper = coord_mapper.CoordMapperCSG(match_code=(left_side_corner_pixels, right_side_corner_pixels))

	global myDetector 
	myDetector = PlayerDetectorDetectron2(leftSideCorners=left_side_corner_pixels, 
											rightSideCorners=right_side_corner_pixels_added_half_field, 
											coordMapper=coordMapper, origResolution=origRes, 
											outResolution=outRes, segnumx=2, segnumy=2, nmsThreshold=0.5)

if __name__ == "__main__":
	initApplication()
	app.run(host='0.0.0.0', port=5555, threaded=False)