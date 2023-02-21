import logging
from enum import Enum

import filterpy.common
import numpy as np

from kalmanfilter.ekf_base import BaseEKFWithSmoother

logger = logging.getLogger(__name__)


class StateIndex(Enum):
    X_REAR = 0  # x position of rear axis
    Y_REAR = 1  # y position of rear axis
    V = 2  # velocity
    PSI = 3  # orientation (yaw angle)
    YAW_RATE = 4  # yaw rate
    LENGTH = 5


class EKFConstVelocityConstYawRate(BaseEKFWithSmoother):
    """
    Extended Kalman Filter class for constant yaw rate model with state = [x,y,v,psi,omega] and measurement = [x_center,y_center]
    """

    def __init__(self, dt, cov_measuring, var_process_acc=0.01, var_process_omega=0.01,
                 l_r=1.5, **kwargs):
        """
        Initializes EKFConstVelocityConstYawrate class

        :param dt: time increment of input data
        :param sigma_pos_x: standard deviation of measurement x position
        :param sigma_pos_y: standard deviation of measurement y position
        :param tau_v: process noise variance for velocity
        :param tau_psi: process noise variance for yaw angle
        :param tau_omega: process noise variance for yaw rate
        :param l_r: distance from the vehicle's center to its rear axis
        """
        super().__init__(dt=dt, dim_x=5, dim_z=2)
        # covariance matrix
        self.P = np.diag([0.1, 0.1, 10., np.deg2rad(4.), np.deg2rad(1.)]).astype(np.float64)
        # measurement noise
        self.R = np.array(cov_measuring)
        if self.R.ndim == 1:
            self.R = np.diag(self.R)
        # process noise (continuous)
        self.Q = np.diag([0, 0, var_process_acc, 0, 0])
        # Orientation noise
        orientation_noise = filterpy.common.Q_discrete_white_noise(2, dt, var_process_omega)
        self.Q[3:, 3:] = orientation_noise
        # distance from center to rear axis
        self.l_r = l_r

    def init_x(self, pos_x: float, pos_y: float, theta: float, vel: float, **kwargs) -> None:
        """
        Set's initial state of filter

        :param pos_x: initial x coordinate of center position
        :param pos_y: initial y coordinate of center position
        :param theta: initial orientation
        :param vel: initial velocity
        """
        # shift positions from center to rear axis
        pos_x_rear = pos_x - self.l_r * np.cos(theta)
        pos_y_rear = pos_y - self.l_r * np.sin(theta)
        self.x = np.array([pos_x_rear, pos_y_rear, vel, theta, 0.0])

    def calculate_F_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates Jacobian F = df/dx of discrete state transition function f, where x[k+1] = f(x[k])

        :param x: state vector
        """

        # Retrieve values form state
        v = x[StateIndex.V.value]
        yaw = x[StateIndex.PSI.value]

        A = np.array([[0, 0, np.cos(yaw), -v * np.sin(yaw), 0],
                      [0, 0, np.sin(yaw), v * np.cos(yaw), 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
        F = np.identity(self.dim_x) + A * self.dt
        return F

    def calculate_H_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        """
        Returns Jacobian dh/dx of the h matrix (measurement function)
        where h = [x_rear + l_r * cos(psi), y_rear + l_r * sin(psi)]

        :param x: state vector
        """

        H = np.array([[1., 0., 0., -self.l_r * np.sin(x[StateIndex.PSI.value]), 0.],
                      [0., 1., 0., self.l_r * np.cos(x[StateIndex.PSI.value]), 0.]])
        return H

    def calculate_hx(self, x: np.ndarray) -> np.ndarray:
        """
        Maps state variable (rear axis) to corresponding measurement z (center)

        :param x: state vector
        """
        x_center = x[StateIndex.X_REAR.value] + self.l_r * np.cos(x[StateIndex.PSI.value])
        y_center = x[StateIndex.Y_REAR.value] + self.l_r * np.sin(x[StateIndex.PSI.value])
        return np.array([x_center, y_center])

    def calculate_H_jacobian_pseudo(self, x):
        H = np.zeros((2, 5))
        H[0, StateIndex.V.value] = 1.
        H[1, StateIndex.YAW_RATE.value] = 1.
        return H

    def calculate_hx_pseudo(self, x):
        return x.ravel()[[StateIndex.V.value, StateIndex.YAW_RATE.value]]

    def update_pseudo_measurement(self, v_pseudo=0, yaw_rate_pseudo=0):
        v = self.get_vel(self.x)
        if np.abs(v) < 1.:
            logger.debug("Velocity small: %.2f m/s! Inserting pseudo measurement!", v)
            # Values initially taken from Egon, adjusted
            sigma_vel = np.fmax(0.5, 3 * np.abs(v))
            sigma_yaw_rate = np.fmax(0.1, 1.6 * np.abs(v))
            R = np.diag([sigma_vel, sigma_yaw_rate])
            z_pseudo = np.array([v_pseudo, yaw_rate_pseudo])
            self.update(z_pseudo, self.calculate_H_jacobian_pseudo, self.calculate_hx_pseudo, R=R)

    def propagate_x(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the next state of X.

        :param x: state vector
        """

        # Get values form state
        # pos_x = x[StateIndex.X_REAR.value]
        # pos_y = x[StateIndex.Y_REAR.value]
        v = x[StateIndex.V.value]
        psi = x[StateIndex.PSI.value]
        yaw_rate = x[StateIndex.YAW_RATE.value]

        # Compute x_dot
        px_dot = v * np.cos(psi)
        py_dot = v * np.sin(psi)
        v_dot = 0
        psi_dot = yaw_rate
        yaw_rate_dot = 0

        x_dot = np.array([px_dot, py_dot, v_dot, psi_dot, yaw_rate_dot])

        # State predict
        x_next = x + self.dt * x_dot
        return x_next

    def get_pos_x_center(self, x):
        """
        Getter that retrieves x-position of model's center from state x.
        """
        if x.shape == (StateIndex.LENGTH.value,):
            return x[StateIndex.X_REAR.value] + self.l_r * np.cos(x[StateIndex.PSI.value])
        else:
            return x[:, StateIndex.X_REAR.value] + self.l_r * np.cos(x[:, StateIndex.PSI.value])

    def get_pos_y_center(self, x):
        """
        Getter that retrieves y-position of model's center from state x.
        """
        if x.shape == (StateIndex.LENGTH.value,):
            return x[StateIndex.Y_REAR.value] + self.l_r * np.sin(x[StateIndex.PSI.value])
        else:
            return x[:, StateIndex.Y_REAR.value] + self.l_r * np.sin(x[:, StateIndex.PSI.value])

    @staticmethod
    def get_vel(x):
        """
        Getter that retrieves velocity from state x.
        """
        if x.shape == (StateIndex.LENGTH.value,):
            return x[StateIndex.V.value]
        else:
            return x[:, StateIndex.V.value]

    @staticmethod
    def get_theta(x):
        """
        Getter that retrieves orientation from state x.
        """
        if x.shape == (StateIndex.LENGTH.value,):
            return x[StateIndex.PSI.value]
        else:
            return x[:, StateIndex.PSI.value]
