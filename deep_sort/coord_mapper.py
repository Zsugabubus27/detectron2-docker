import cv2
import numpy as np

ISSIA_vendeg_elorol = {
    'originalAnchors' : [(989, 103), (1644, 100), (1875, 396), (1007, 398)],
    'transformedAnchors' : [(165, 0), (0, 0), (0, 334.5), (165, 334.5)]
}

ISSIA_kozep_elorol = {
    'originalAnchors' : [(944,102), (1441, 399), (944, 520), (448, 399)],
    'transformedAnchors' : [(543.5, 0.0), (543.5-91.5, 334.5), (543.5, 334.5+91.5), (543.5+91.5, 334.5)]
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


class CoordMapperCSG():
    def __init__(self):
        pts_on_original_left = [[1264.7023,499.5309],[1870.1519,549.1178 ],[2517.0312,1275.4386 ],[ 192.16057,548.8432 ]]
        pts_on_original_right = [[755.81616,474.7149],[1361.8402,456.584],[2482.6863,568.66864],[257.55048,1084.8362]]
        self.pts_on_transformed = [[0,0],[521,0],[521,661],[0,661]]
        self.full_size_width = 2560 * 2
        self.full_size_height = 1440 * 2

        # Transzformációs mátrixok
        self.M_left = cv2.getPerspectiveTransform(np.float32(pts_on_original_left),np.float32(self.pts_on_transformed))
        self.M_left_inv = np.linalg.pinv(self.M_left)
        self.M_right = cv2.getPerspectiveTransform(np.float32(pts_on_original_right),np.float32(self.pts_on_transformed))
        self.M_right_inv = np.linalg.pinv(self.M_right)

    def image2xy(self, points):
        """
        Kép koordinátarendszerből valós koordinátarendszerbe konvertálja a bemenetet.
        """
        if len(points) == 0:
            return np.array([[]])
        # pointsArray = [[x if x < (self.full_size_width / 2) else, y] for x, y in points]
        pointsArray = np.float32(points)
        transformedPoints = []
        for x, y in pointsArray:
            if x < self.full_size_width / 2:
                # BAL oldali pontok
                x_transf, y_transf = cv2.perspectiveTransform(np.array([[[x, y]]]), self.M_left)[0][0]
                # Ha bal oldali kamera észlel jobb oldali játékost
                if x_transf > self.pts_on_transformed[1][0]:
                    transformedPoints.append(None)
                    print(x, y)
                else:
                    transformedPoints.append([x_transf, y_transf])
            else:
                # JOBB oldali pontok
                toTransform = np.array([[[ x - int(self.full_size_width / 2),    y   ]]])
                x_transf, y_transf = cv2.perspectiveTransform(toTransform, self.M_right)[0][0]
                x_transf += self.pts_on_transformed[1][0] - self.pts_on_transformed[0][0]
                # Ha bal oldali kamera észlel jobb oldali játékost
                if x_transf < self.pts_on_transformed[1][0]:
                    transformedPoints.append(None)
                    print(x, y)
                else:
                    transformedPoints.append([x_transf, y_transf])

        if transformedPoints == []:
            return np.array([[]])
        else:
            return transformedPoints

    def xy2image(self, points):
        """
        Valós koordinátarendszerből kép koordinátarendszerbe konvertálja a bemenetet.
        """
        if len(points) == 0:
            return np.array([[]])
        pointsArray = np.float32(points)
        transformedPoints = []
        transformed_width = self.pts_on_transformed[1][0] - self.pts_on_transformed[0][0]
        for x, y in pointsArray:
            if x < transformed_width:
                # Baloldali vissza trafó
                x_transf, y_transf = cv2.perspectiveTransform(np.array([[[x, y]]]), self.M_left_inv)[0][0]
                transformedPoints.append([x_transf, y_transf])
            else:
                # Jobboldali visszatrafó
                toTransform = np.array([[[ x - int(transformed_width),   y  ]]])
                x_transf, y_transf = cv2.perspectiveTransform(toTransform, self.M_right_inv)[0][0]
                x_transf += self.full_size_width / 2
                transformedPoints.append([x_transf, y_transf])

        if transformedPoints == []:
            return np.array([[]])
        else:
            return transformedPoints

if __name__ == '__main__':
    cm = CoordMapperCSG()
    transformed = cm.image2xy([[3249,529], [2946, 554], [2077, 619], (3403, 524), [1391, 550]])
    print(transformed)
    retrafo = cm.xy2image([x for x in transformed if x is not None])
    print(retrafo)
