import cv2
import numpy as np

ISSIA_vendeg_elorol = {
    'originalAnchors' : [(989, 103), (1644, 100), (1875, 396), (1007, 398)],
    'transformedAnchors' : [(165, 0), (0, 0), (0, 334.5), (165, 334.5)]
}

class CoordMapper():
    def __init__(self, anchorDict):
        pts_on_original = anchorDict['originalAnchors']
        pts_on_transformed = anchorDict['transformedAnchors']
        # Transzformációs mátrix
        self.M = cv2.getPerspectiveTransform(np.float32(pts_on_original),np.float32(pts_on_transformed))
        self.M_inv = np.linalg.pinv(self.M)

    def image2xy(self, points):
        """
        Kép koordinátarendszerből valós koordinátarendszerbe konvertálja a bemenetet.
        """
        if len(points) == 0:
            return np.array([[]])
        pointsArray = np.float32([points])
        transformedPoints = cv2.perspectiveTransform(pointsArray, self.M)
        if transformedPoints is None:
            return np.array([[]])
        else:
            return transformedPoints[0]

    def xy2image(self, points):
        """
        Valós koordinátarendszerből kép koordinátarendszerbe konvertálja a bemenetet.
        """
        if len(points) == 0:
            return np.array([[]])
        pointsArray = np.float32([points])
        transformedPoints = cv2.perspectiveTransform(pointsArray, self.M_inv)
        if transformedPoints is None:
            return np.array([[]])
        else:
            return transformedPoints[0]