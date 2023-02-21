import logging
from abc import abstractmethod, ABCMeta

import filterpy.kalman
import numpy as np

logger = logging.getLogger(__name__)


class BaseEKFWithSmoother(filterpy.kalman.ExtendedKalmanFilter, metaclass=ABCMeta):
    """
    Base Extended Kalman Filter class, that extends ExtendedKalmanFilter class of filterpy package.
    """

    def __init__(self, dt, dim_x, dim_z):
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.dt = dt

    @abstractmethod
    def init_x(self, pos_x: float, pos_y: float, theta: float, vel: float) -> None:
        """
        Initializes state
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_F_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates Jacobian F = df/dx of discrete state transition function f, where x[k+1] = f(x[k])
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_H_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        """
        Returns Jacobian dh/dx of the h matrix (measurement function)
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_hx(self, x: np.ndarray) -> np.ndarray:
        """
        Maps state variable to corresponding measurement z
        """
        raise NotImplementedError()

    @abstractmethod
    def propagate_x(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the next state of X.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_pos_x_center(self, x):
        """
        Getter that retrieves x-position of model's center from state x.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_pos_y_center(self, x):
        """
        Getter that retrieves y-position of model's center from state x.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_vel(x):
        """
        Getter that retrieves velocity from state x.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_theta(x):
        """
        Getter that retrieves orientation from state x.
        """
        raise NotImplementedError()

    def update_from_measurement(self, z, measurement_noise=None):
        """
        Updates the kalman filter state matrices from measurement z
        """
        self.update(z, self.calculate_H_jacobian_at, self.calculate_hx, R=measurement_noise)

    def predict_x(self, u=0):
        """
        Predicts the next state of X.
        """
        self.F = self.calculate_F_jacobian_at(self.x)
        x_next = self.propagate_x(self.x)
        self.x = x_next

    def rts_smoother(self, Xs, Ps, inv=np.linalg.inv):
        """
        Reference: Bayesian Filtering and Smoothing,
        see https://www.cambridge.org/core/books/bayesian-filtering-and-smoothing/C372FB31C5D9A100F8476C1B23721A67
        Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by an Extend Kalman filter.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        inv : function, default numpy.linalg.inv
            If you prefer another inverse function, such as the Moore-Penrose
            pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv

        Returns
        -------

        x : numpy.ndarray
           smoothed means

        G : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Pp : numpy.ndarray
           Predicted state covariances

        """

        if len(Xs) != len(Ps):
            raise ValueError('length of Xs and Ps must be the same')

        n = Xs.shape[0]
        dim_x = Xs.shape[1]

        # smoother gain
        G = np.zeros((n, dim_x, dim_x))

        x, xp, P, Pp = Xs.copy(), Xs.copy(), Ps.copy(), Ps.copy()
        for k in range(n - 2, -1, -1):
            F = self.calculate_F_jacobian_at(x[k])
            # predicted mean
            # xp[k+1] = np.dot(F, x[k])
            xp[k + 1] = self.propagate_x(x[k])  # nonlinear equation
            # predicted covariance
            Pp[k + 1] = np.dot(np.dot(F, P[k]), F.T) + self.Q

            # Gain matrix
            G[k] = np.dot(np.dot(P[k], F.T), inv(Pp[k + 1]))
            # smoothed mean
            x[k] = x[k] + np.dot(G[k], x[k + 1] - xp[k + 1])
            # smoothed covariance
            P[k] = P[k] + np.dot(np.dot(G[k], P[k + 1] - Pp[k + 1]), G[k].T)

        return x, P, G, Pp

    def calculate_H_jacobian_pseudo(self, x):
        pass

    def calculate_hx_pseudo(self, x):
        pass

    def update_pseudo_measurement(self, v_pseudo=0, yaw_rate_pseudo=0):
        logger.warning("Pseudo measurement not implemented!")
