# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
import pandas as pd

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = { 1: 3.8415, 2: 5.9915, 3: 7.8147,
             4: 9.4877, 5: 11.070, 6: 12.592,
             7: 14.067, 8: 15.507, 9: 16.919}


class KalmanFilterWorldCoordinate(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the position of the player's foot (x, y)  And the aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The player's foot
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        # _motion_mat: Ez az A mátrix: ebben van elkódolva a lineáris mozgásmodell
        # _update_mat: Ez a H mx
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        # TODO: Change path according to input
        self.errorDf = pd.read_csv('/home/dobreff/work/Dipterv/MLSA20/vendeg_elorol_position_error_binned.csv')


        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        # self._std_weight_position = 1. / 20
        # self._std_weight_velocity = 1. / 160
    def getError(self, x, y):
        xbin = round(x / 25)
        ybin = round(y / 25)
        retDf = self.errorDf[(self.errorDf.xbin == xbin) & (self.errorDf.ybin == ybin)]

        if len(retDf) == 1:
            return retDf[['error', 'height_error', 'aspectRatio_error']].iloc[0].values
        else:
            retDf = self.errorDf[(self.errorDf.xbin.isin([xbin, xbin - 1, xbin + 1])) & (self.errorDf.ybin.isin([ybin, ybin - 1, ybin + 1]))]
            if len(retDf) < 1:
                return self.errorDf[['error', 'height_error', 'aspectRatio_error']].mean().values
            else:
                return retDf[['error', 'height_error', 'aspectRatio_error']].mean().values
        #return self.errorDf[(self.errorDf.xbin == xbin) & (self.errorDf.ybin == ybin)].error.iloc[0]

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        # Mean vektor itt a 4 measurement értéket tartalmazza és 4 nullát a sebességeknek
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Ez a process kovariancia mx átlója, azaz ezek az értékek a megfelelő varianciák
        # TODO: Ez a P, tehát a kezdeti process covariance. Erre valami jó becslést kell adni.
        posError, heightError, aspectError = self.getError(measurement[0], measurement[1])
        std = [
            posError,
            posError,
            aspectError * 2.5,
            heightError * 2.5,
            posError / 25,
            posError / 25,
            aspectError / 25,
            heightError / 25]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # Ez a Qk-nak a kiszámolása. Qk függ a BBox magasságától
        # TODO: Process covariance mx, valahogy kimérni ezt is
        posError, heightError, aspectError = self.getError(mean[0], mean[1])
        Qk = [
            posError,
            posError,
            aspectError * 2.5,
            heightError * 2.5,
            posError / 25,
            posError / 25,
            aspectError / 25,
            heightError / 25]
        motion_cov = np.diag(np.square(Qk))

        # A*X_{k-1}-nek felel meg: Tehát kiszámolja az előző állapot és a motion model alapján a kövi állapotot
        mean = np.dot(self._motion_mat, mean)
        # Kiszámolja az előző állapot covarianciájából: A*P_{k-1}*A^T + Qk
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """

        posError, heightError, aspectError = self.getError(mean[0], mean[1])
        std = [
            posError,
            posError,
            aspectError * 2.5,
            heightError * 2.5]
        # TODO: Ez az R mx, ami a mérési pontatlanságot tartalmazza:
        # dx, dy, da, dh --> távolság függvényében kell ezeknek szerepelniük!!!
        # Ez az R mx
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))

        # Visszatérési érték: H*Xkp, H*Pkp*H^T + R
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # H*Xkp, H*Pkp*H^T + R
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kiszámolja a K mátrixot (4x4)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # innovation = [Y - H*Xkp]
        innovation = measurement - projected_mean

        # Xk : new state
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # Pk : new process covariance mx
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        # H*Xk, H*Pk*H^T + R
        # Azaz: Az állapotnak csak az első 4 értéke (x, y, dx, dy)
        # És az állapot kov.mx-nak is csak a bal felső 4x4-es mx-a
        mean, covariance = self.project(mean, covariance)
        
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        #print('KF', mean, measurements, covariance)
        # with open('kalmanFilter.txt', 'a') as fd:
        #     fd.write('{};{};{};{}\n'.format(*mean[:2], *measurements[0,:2]))

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
