from enum import Enum

import filterpy.common
import numpy as np

from kalmanfilter.ekf_base import BaseEKFWithSmoother


class StateIndex(Enum):
    X_CENTER = 0  # x position of center of mass
    Y_CENTER = 1  # y position of center of mass
    VX = 2  # velocity in x direction
    VY = 3  # velocity in y direction
    LENGTH = 4


class EKFPointMass(BaseEKFWithSmoother):
    """
    Extended Kalman Filter class for Point Mass Model with state = [x,y,vx,vy] and measurement = [x_center,y_center]
    """

    def __init__(self, dt, var_meas_pos_x=0.5, var_meas_pos_y=0.5, var_process_v=0.1, **kwargs):
        """
        Initializes EKFPointMass class

        :param dt: time increment of input data
        :param sigma_pos_x: standard deviation of measurement x position
        :param sigma_pos_y: standard deviation of measurement y position
        :param tau_vx: process noise variance for x velocity
        :param tau_vy: process noise variance y velocity
        """
        super().__init__(dt=dt, dim_x=4, dim_z=2)
        # state uncertainty
        self.R0 = np.diag([var_meas_pos_x, var_meas_pos_y])
        self.R = self.R0
        # process uncertainty
        self.Q = filterpy.common.Q_discrete_white_noise(2, dt=self.dt, var=var_process_v, block_size=2,
                                                        order_by_dim=False)
        # covariance matrix
        self.P = np.diag([var_meas_pos_x, var_meas_pos_y, var_process_v, var_process_v])
        self.x_dot = filterpy.common.kinematic_kf(dim=2, order=1, dt=self.dt, order_by_dim=False).F

    def init_x(self, pos_x: float, pos_y: float, theta: float, vel: float, **kwargs) -> None:
        """
        Set's initial state of filter

        :param pos_x: initial x coordinate of center position
        :param pos_y: initial y coordinate of center position
        :param theta: initial orientation
        :param vel: initial velocity
        """
        self.x = np.array([pos_x, pos_y, vel * np.cos(theta), vel * np.sin(theta)])

    def calculate_F_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates Jacobian F = df/dx of discrete state transition function f, where x[k+1] = f(x[k])

        :param x: state vector
        """
        return self.x_dot

    def calculate_H_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        """
        Returns Jacobian dh/dx of the h matrix (measurement function) where h = [x, y]

        :param x: state vector
        """
        return np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])

    def calculate_hx(self, x: np.ndarray) -> np.ndarray:
        """
        Maps state variable to corresponding measurement z

        :param x: state vector
        """
        return np.array([x[StateIndex.X_CENTER.value], x[StateIndex.Y_CENTER.value]])

    def propagate_x(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the next state of X.

        :param x: state vector
        """

        x_next = np.dot(self.F, x)
        return x_next

    def get_pos_x_center(self, x):
        """
        Getter that retrieves x-position of model's center from state x.
        """
        if x.shape == (StateIndex.LENGTH.value,):
            return x[StateIndex.X_CENTER.value]
        else:
            return x[:, StateIndex.X_CENTER.value]

    def get_pos_y_center(self, x):
        """
        Getter that retrieves y-position of model's center from state x.
        """
        if x.shape == (StateIndex.LENGTH.value,):
            return x[StateIndex.Y_CENTER.value]
        else:
            return x[:, StateIndex.Y_CENTER.value]

    @staticmethod
    def get_vel(x):
        """
        Getter that retrieves velocity from state x.
        """
        if x.shape == (StateIndex.LENGTH.value,):
            vx = x[StateIndex.VX.value]
            vy = x[StateIndex.VY.value]
        else:
            vx = x[:, StateIndex.VX.value]
            vy = x[:, StateIndex.VY.value]
        vel = np.sqrt(pow(vx, 2) + pow(vy, 2))
        return vel

    @staticmethod
    def get_theta(x):
        """
        Getter that retrieves orientation from state x.
        """
        if x.shape == (StateIndex.LENGTH.value,):
            vx = x[StateIndex.VX.value]
            vy = x[StateIndex.VY.value]
        else:
            vx = x[:, StateIndex.VX.value]
            vy = x[:, StateIndex.VY.value]
        theta = np.arctan2(vy, vx)
        return theta
