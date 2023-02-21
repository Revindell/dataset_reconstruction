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
    ACC = 5  # acceleration
    LENGTH = 6  # length


class EKFConstAccelerationConstTurnRate(BaseEKFWithSmoother):
    """
    Extended Kalman Filter class for constant acceleration and turn rate with state = [x,y,v,psi,omega,acc] and measurement = [x_center,y_center]
    """

    def __init__(self, dt, cov_measuring, var_process_acc=0.01, var_process_omega=0.01,
                 l_r=1.5, **kwargs):
        """
        Initializes EKFConstVelocityConstYaw-rate class

        :param dt: time increment of input data
        :param sigma_pos_x: standard deviation of measurement x position
        :param sigma_pos_y: standard deviation of measurement y position
        :param tau_v: process noise variance for velocity
        :param tau_psi: process noise variance for yaw angle
        :param tau_omega: process noise variance for yaw rate
        :param tau_acc: process noise variance for acceleration
        :param l_r: distance from the vehicle's center to its rear axis
        """
        super().__init__(dt=dt, dim_x=6, dim_z=2)
        # covariance matrix
        self.P = np.diag([1, 1, 10., np.deg2rad(4.), np.deg2rad(1.), 0.1]).astype(np.float64)
        # measurement noise
        self.R = np.array(cov_measuring)
        if self.R.ndim == 1:
            self.R = np.diag(self.R)
        # default process noise (continuous)
        self.Q = np.diag([0, 0, var_process_acc, 0, 0, var_process_acc])
        # Orientation noise
        orientation_noise = filterpy.common.Q_discrete_white_noise(2, dt, var_process_omega)
        self.Q[3:5, 3:5] = orientation_noise
        # distance from center to rear axis
        self.l_r = l_r

    def init_x(self, pos_x: float, pos_y: float, theta: float, vel: float, omega: float = 0.0,
               acc: float = 0.0) -> None:
        """
        Set's initial state of filter

        :param pos_x: initial x coordinate of center position
        :param pos_y: initial y coordinate of center position
        :param theta: initial orientation
        :param vel: initial velocity
        :param omega: initial yaw rate
        :param acc: initial acceleration
        """
        # shift positions from center to rear axis
        pos_x_rear = pos_x - self.l_r * np.cos(theta)
        pos_y_rear = pos_y - self.l_r * np.sin(theta)
        self.x = np.array([pos_x_rear, pos_y_rear, vel, theta, omega, acc])

    def calculate_F_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates Jacobian F = df/dx of discrete state transition function f, where x[k+1] = f(x[k])

        :param x: state vector
        """

        # Retrieve values form state
        v = x[StateIndex.V.value]
        yaw = x[StateIndex.PSI.value]

        A = np.array([[0, 0, np.cos(yaw), -v * np.sin(yaw), 0, 0],
                      [0, 0, np.sin(yaw), v * np.cos(yaw), 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        F = np.identity(self.dim_x) + A * self.dt
        return F

    def calculate_H_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        """
        Returns Jacobian dh/dx of the h matrix (measurement function)
        where h = [x_rear + l_r * cos(psi), y_rear + l_r * sin(psi)]

        :param x: state vector
        """

        H = np.array([[1., 0., 0., -self.l_r * np.sin(x[StateIndex.PSI.value]), 0., 0.],
                      [0., 1., 0., self.l_r * np.cos(x[StateIndex.PSI.value]), 0., 0.]])

        return H

    def calculate_hx(self, x: np.ndarray) -> np.ndarray:
        """
        Maps state variable (rear axis) to corresponding measurement z (center)

        :param x: state vector
        """
        x_center = x[StateIndex.X_REAR.value] + self.l_r * np.cos(x[StateIndex.PSI.value])
        y_center = x[StateIndex.Y_REAR.value] + self.l_r * np.sin(x[StateIndex.PSI.value])
        return np.array([x_center, y_center])

    def propagate_x(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the next state of X

        :param x: state vector
        """

        # Get values form state
        pos_x = x[StateIndex.X_REAR.value]
        pos_y = x[StateIndex.Y_REAR.value]
        v = x[StateIndex.V.value]
        psi = x[StateIndex.PSI.value]
        yaw_rate = x[StateIndex.YAW_RATE.value]
        acc = x[StateIndex.ACC.value]

        # Compute x_dot
        # px_dot = v * np.cos(psi)
        # py_dot = v * np.sin(psi)
        v_dot = acc
        psi_dot = yaw_rate
        # yaw_rate_dot = 0
        # acc_dot = 0

        v_next = v + v_dot * self.dt
        psi_next = psi + psi_dot * self.dt
        if yaw_rate > 0.1:
            px_next = pos_x + (v_next * yaw_rate * np.sin(psi_next) + acc * np.cos(psi_next) - v * yaw_rate * np.sin(
                psi) - acc * np.cos(psi)) / (yaw_rate ** 2)
            py_next = pos_y - (v_next * yaw_rate * np.cos(psi_next) - acc * np.sin(psi_next) - v * yaw_rate * np.cos(
                psi) + acc * np.sin(psi)) / (yaw_rate ** 2)
        else:
            px_next = pos_x + (v * self.dt + acc * self.dt ** 2 / 2) * np.cos(psi)
            py_next = pos_y + (v * self.dt + acc * self.dt ** 2 / 2) * np.sin(psi)

        X_next = np.array([px_next, py_next, v_next, psi_next, yaw_rate, acc])

        return X_next

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
