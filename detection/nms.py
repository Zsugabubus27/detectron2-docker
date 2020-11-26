#needs parameters this way
def boxMaker(x,y,w,h):
	'''
	Convert top left x,y,w,h to (tlXY, trXY, brXY, blXY)
	'''
	rectangle = ((x,y),(x+w,y),(x+w,y+h),(x,y+h))
	return rectangle


def overlap(boxA,boxB,threshold):

	x1 = max(boxA[0][0], boxB[0][0]) #TL pontok közül a leg jobb lentebbi
	y1 = max(boxA[0][1], boxB[0][1]) #TL pontok közül a leg jobb lentebbi
	x2 = min(boxA[2][0], boxB[2][0]) #BR pontok közül a leg bal fentebbi
	y2 = min(boxA[2][1], boxB[2][1]) #BR pontok közül a leg bal fentebbi
	# Intersection area Width és Height kiszámítása: Ha nem metszenek akkor kinullázom
	w = max(0, x2 - x1 + 1)
	h = max(0, y2 - y1 + 1)

	intersection_area = float(w*h)
	area_boxA = (boxA[2][0]-boxA[0][0]+1) * (boxA[2][1]-boxA[0][1]+1)
	area_boxB = (boxB[2][0]-boxB[0][0]+1) * (boxB[2][1]-boxB[0][1]+1)
	union_area = (area_boxA) + (area_boxB) - intersection_area

	value = float(intersection_area) / float(union_area)
	return (value > threshold)
	
def nms(bboxList,threshold):
	'''
	bboxList : List of bounding boxes (top left x,y,w,h)
	threshold : float

	Results:
		result_list : List of bounding boxes after non-maxima supression
		bools : Mask list for the original list
	'''
	bools = [True] * len(bboxList)
	
	for l, t in [(x, y) for x in range(len(bboxList)) for y in range(len(bboxList)) if x>y]:    
		bboxA = boxMaker(*bboxList[l])
		bboxB = boxMaker(*bboxList[t])
		if(overlap(bboxA, bboxB, threshold)):
			bools[l] = False

	result_list = [elem for elem, b in zip(bboxList, bools) if b]
	return result_list, bools
		
	
if __name__ == '__main__':
	bbs = [(12, 84, 140, 212),
	(24, 84, 152, 212),
	(36, 84, 164, 212),
	(12, 96, 140, 224),
	(24, 96, 152, 224),
	(24, 108, 152, 236)]

	# Input: X,Y,W,H
	list_ = [(startX, startY, endX-startX, endY-startY)for startX, startY, endX, endY in bbs]
	threshold = 0.5
	result = nms(list_,threshold)

	print('-'*25)
	testDetectedPlayers = [{'x': 1130, 'y': 561, 'w': 26, 'h': 39, 'color': 'yellow'}, {'x': 1607, 'y': 552, 'w': 11, 'h': 15, 'color': 'yellow'}, {'x': 1606, 'y': 533, 'w': 15, 'h': 19, 'color': 'yellow'}, {'x': 1871, 'y': 542, 'w': 41, 'h': 63, 'color': 'red'}, {'x': 1745, 'y': 540, 'w': 28, 'h': 29, 'color': 'yellow'}, {'x': 1130, 'y': 561, 'w': 26, 'h': 39, 'color': 'yellow'}, {'x': 1607, 'y': 552, 'w': 11, 'h': 15, 'color': 'yellow'}, {'x': 1871, 'y': 542, 'w': 41, 'h': 63, 'color': 'red'}, {'x': 1745, 'y': 540, 'w': 28, 'h': 29, 'color': 'yellow'}, {'x': 1606, 'y': 533, 'w': 15, 'h': 19, 'color': 'yellow'}, {'x': 1130, 'y': 561, 'w': 26, 'h': 39, 'color': 'yellow'}, {'x': 1606, 'y': 533, 'w': 15, 'h': 19, 'color': 'yellow'}, {'x': 1871, 'y': 560, 'w': 39, 'h': 45, 'color': 'red'}]
	nms([(p['x'], p['y'], p['w'], p['h']) for p in testDetectedPlayers ], threshold=0.7)
	

	
