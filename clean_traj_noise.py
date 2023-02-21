import logging
from enum import Enum
from typing import Tuple

import pandas as pd
import seaborn as sns
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle
from commonroad.scenario.trajectory import State, Trajectory
from matplotlib import pyplot as plt

from kalmanfilter.EM_estimate_process_noise import estimate_process_noise
from kalmanfilter.efk_const_acceleration_const_turnrate import EKFConstAccelerationConstTurnRate
from kalmanfilter.ekf_base import BaseEKFWithSmoother
from kalmanfilter.ekf_const_velocity_const_yawrate import EKFConstVelocityConstYawRate
from kalmanfilter.ekf_point_mass import EKFPointMass
from strategy_interface import *

logger = logging.getLogger(__name__)


class KFModelChoices(str, Enum):
    """ Enum for the different KF models.
    """
    EKFConstVelocityConstYawRate = 'ConstVelocityConstYawRate'
    EKFConstAccelerationConstTurnRate = 'ConstAccelerationConstTurnRate'
    EKFPointMass = 'PointMass'


class CleanTrajNoise(Strategy):
    """ Class which tackles the problem of trajectory noise in the Argoverse dataset (converted into CommonRoad
    Scenario). A Concrete Strategy in the Strategy Pattern Design.
    """

    def clean_scenario(self, scenario: Scenario, model_choice: KFModelChoices = None,
                       measurement_noise: np.ndarray = None, process_noise: np.ndarray = None, **kwargs) -> Scenario:
        """ Cleans the dataset from trajectory noise. It overrides the abstract method from the abstract class Strategy.

        :param scenario: scenario before cleansing
        :param model_choice: choice of the kalman filter model to be used
        :param measurement_noise: measurement noise of the kalman filter
        :param process_noise: process noise of the kalman filter
        :return: scenario after cleansing
        """
        # set the model choice to the default value if it is not set
        new_scenario = scenario

        if model_choice is None:
            model_choice = [KFModelChoices.EKFConstAccelerationConstTurnRate]
        # get the sampling time dt from the scenario, default is 0.04
        if scenario.dt is not None:
            dt = scenario.dt
        else:
            dt = 0.04

        # measurement covariance matrix
        if measurement_noise is not None:
            R = measurement_noise
        else:
            R = np.diag([30, 30])

        # calculating Q matrix using EM-algorithm
        if KFModelChoices.EKFConstAccelerationConstTurnRate in model_choice:
            efk_model = EKFConstAccelerationConstTurnRate(dt, R)
            process_noise_CACTR = estimate_process_noise(new_scenario, efk_model, num_iterations=20, num_init_steps=15)
        else:
            process_noise_CACTR = None

        for obs in new_scenario.dynamic_obstacles:
            # remove obstacle from the scenario
            new_scenario.remove_obstacle(obs)
            # check the existing time of the dynamic obstacles, if it's too short, remove it!
            obs_statelist = []
            obs_statelist = self.create_state_list(obs)
            # overlook the dynamic obstacles with too short time
            if len(obs_statelist) < 20:
                continue

            # create a measurement simulator for the EKF model
            simulator = Simulator(obs=obs, dt=dt, pos_x=0., pos_y=0., theta=0., vel=0., acc=0.0, omega=0.0)
            # create the EKF model
            for model in model_choice:
                if model == 'ConstVelocityConstYawRate':
                    simulator.register_filter(EKFConstVelocityConstYawRate(dt=dt, cov_measuring=R),
                                              "ConstVelocityConstYawRate")
                    if process_noise is not None:
                        simulator.filters["ConstAccelerationConstTurnRate"].Q = process_noise
                if model == 'ConstAccelerationConstTurnRate':
                    simulator.register_filter(EKFConstAccelerationConstTurnRate(dt=dt, cov_measuring=R),
                                              "ConstAccelerationConstTurnRate")
                    # setting the process noise of the EKF model
                    simulator.filters["ConstAccelerationConstTurnRate"].Q = process_noise_CACTR
                if model == 'PointMass':
                    simulator.register_filter(EKFPointMass(dt=dt, cov_measuring=R), "PointMass")
                    if process_noise is not None:
                        simulator.filters["PointMass"].Q = process_noise

            # get the smoothed trajectory
            new_track = simulator.run_filtering(err_pos_x=0, err_pos_y=0, err_theta=0, err_vel=0, dt=dt, R=R)
            new_statelist = new_track[new_track['name'] == model][['x', 'y', 'vel', 'psi']]
            new_statelist = new_statelist.values.tolist()

            # using the generated trajectory, create a new dynamic obstacle
            new_dynamic_obstacle = self.create_dynamic_obstacle(new_statelist, obs_statelist, obs)

            # add the new obstacle to the scenario
            new_scenario.add_objects(new_dynamic_obstacle)

        return new_scenario

    @staticmethod
    def create_dynamic_obstacle(new_statelist: np.ndarray, obs_statelist: np.ndarray, obstacle: DynamicObstacle) \
            -> DynamicObstacle:
        """ Creates  new obstacle that has smoothed trajectory generated from Kalman Filter used.

        :param new_statelist：list of states generated from Kalman Filter
        :param obs_statelist：list of states of the original obstacle
        :param obstacle：original obstacle
        :return: new dynamic obstacle
        """
        # get the time steps for the trajectory
        time_steps = obs_statelist[:, 4]
        initial_time_step = int(time_steps[0])

        # create  new state list for the new dynamic obstacle
        new_state_list_converted = []
        for count, value in enumerate(new_statelist):
            position = np.array([value[0], value[1]])
            velocity = value[2]
            orientation = value[3]
            new_state = State(position=position, velocity=velocity, orientation=orientation,
                              time_step=initial_time_step + count)
            new_state_list_converted.append(new_state)

        # create the trajectory of the obstacle, starting at time step 1
        dynamic_obstacle_trajectory = Trajectory(initial_time_step, new_state_list_converted)

        # create the prediction using the trajectory and the shape of the obstacle
        dynamic_obstacle_shape = obstacle.obstacle_shape
        dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)
        # crate initial state
        dynamic_obstacle_initial_state = State(position=[new_statelist[0][0], new_statelist[0][1]],
                                               orientation=new_statelist[0][3],
                                               velocity=new_statelist[0][2], time_step=initial_time_step)
        # generate the dynamic obstacle according to the specification
        dynamic_obstacle_id = obstacle.obstacle_id
        dynamic_obstacle_type = ObstacleType.CAR
        new_dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                               dynamic_obstacle_type,
                                               dynamic_obstacle_shape,
                                               dynamic_obstacle_initial_state,
                                               dynamic_obstacle_prediction)
        return new_dynamic_obstacle


class StateIndex(Enum):
    X = 0
    Y = 1
    V = 2
    PSI = 3


class Simulator:
    """ Simulates the position detection.
    """

    def __init__(self, obs: Scenario.obstacles, dt: float, pos_x: float, pos_y: float, theta: float, vel: float,
                 omega: float = 0., acc: float = 0.) -> None:
        # create thr initial state for the EKF
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.theta = theta
        self.vel = vel
        self.omega = omega
        self.acc = acc
        self.dt = dt
        self.filters = {}
        self.obstacles = obs

    def get_new_measurement(self) -> Tuple[np.ndarray, int]:
        """ Returns measured position. Call once
        for each new measurement at dt time from last call.

        :return: new measurements in form of ([pos_x, pos_y], idx)
        """
        # get the state list of the obstacle
        state_list = []
        for state in self.obstacles.prediction.trajectory.state_list:
            new_state = np.array([state.__getattribute__("position")[0],
                                  state.__getattribute__("position")[1],
                                  state.__getattribute__("velocity"),
                                  state.__getattribute__("orientation")])
            state_list.append(new_state)
        state_list = np.array(state_list).squeeze()

        state_list_length, _ = state_list.shape
        for i in range(state_list_length):
            # get the new position
            self.pos_x = state_list[i, StateIndex.X.value]
            self.pos_y = state_list[i, StateIndex.Y.value]

            # get the new velocity
            self.vel = state_list[i, StateIndex.V.value]

            # get the new orientation
            self.theta = state_list[i, StateIndex.PSI.value]

            # get the new measurement
            meas_pos_x = self.pos_x  # + 0.5 * randn()
            meas_pos_y = self.pos_y  # + 0.5 * randn()
            z = np.array([meas_pos_x, meas_pos_y])
            # yield the measurement
            yield z, i

    def register_filter(self, filters: BaseEKFWithSmoother, name: str) -> None:
        """ Registers a filter to the simulator.

        :param filters: filter to be registered
        :param name: name of the filter
        :return: None
        """
        self.filters[name] = filters

    def plot_results(self, df: pd.DataFrame) -> None:
        """ Plots the results of the simulation.

        :param df: dataframe with the results of the simulation
        :return: None
        """
        # plot the results
        plt.close('all')

        # plot the trajectory
        p = sns.lineplot(data=df, x="time", y="x", hue="name")
        plt.title(self.obstacles.obstacle_id)
        plt.show()
        p = sns.lineplot(data=df, x="time", y="y", hue="name")
        plt.title(self.obstacles.obstacle_id)

        plt.show()

        # plot the orientation
        sns.lineplot(data=df, x="time", y="psi", hue="name")
        plt.title(self.obstacles.obstacle_id)
        plt.show()

        # plot the velocity
        # sns.lineplot(data=df, x="time", y="vel", hue="name")
        # plt.title(self.obstacles.obstacle_id)
        # plt.show()

    def run_filtering(self, err_pos_x: float, err_pos_y: float, err_theta: float, err_vel: float, dt: float,
                      R: np.ndarray) -> pd.DataFrame:
        """ Runs the filtering of the scenario.
        Parameters
        ----------
        err_pos_x: error in the x position
        err_pos_y: error in the y position
        err_theta: error in the orientation
        err_vel: error in the velocity
        dt: time step
        R: covariance matrix

        Returns
        -------
        df: dataframe with the results of the simulation
        """

        for filter_key in self.filters:
            # get the initial state
            initialstate = self.obstacles.prediction.trajectory.state_list[0]
            # get the initial position
            self.pos_x = initialstate.__getattribute__("position")[0]
            self.pos_y = initialstate.__getattribute__("position")[1]
            # get the initial orientation
            self.theta = initialstate.__getattribute__("orientation")
            # get the initial velocity
            self.vel = initialstate.__getattribute__("velocity")
            # make an imperfect starting guess
            self.filters[filter_key].init_x(pos_x=self.pos_x + err_pos_x,
                                            pos_y=self.pos_y + err_pos_y,
                                            theta=self.theta + err_theta,
                                            vel=self.vel + err_vel,
                                            omega=self.omega,
                                            acc=self.acc)

        df = pd.DataFrame()
        # run the filters
        new_measurement = self.get_new_measurement()
        while True:
            # get the next measurement
            z, i = next(new_measurement, [None, None])
            # if no measurement is available, break
            if z is None:
                break
            # get the new state
            new_row = {'name': "input_measurement",
                       'obs_id': self.obstacles.obstacle_id,
                       'x': z[0],
                       'y': z[1],
                       'psi': self.theta,
                       'vel': self.vel,
                       'time': i * dt}
            df = df._append(new_row, ignore_index=True)

            for filter_key in self.filters:
                self.filters[filter_key].update_from_measurement(z, measurement_noise=R)
                self.filters[filter_key].predict()
                # get the new state
                x = self.filters[filter_key].x

                # add the new row to the dataframe
                new_row = {'name': filter_key,
                           'obs_id': self.obstacles.obstacle_id,
                           'x': self.filters[filter_key].get_pos_x_center(x),
                           'y': self.filters[filter_key].get_pos_y_center(x),
                           'psi': self.filters[filter_key].get_theta(x),
                           'vel': self.filters[filter_key].get_vel(x),
                           'time': i * dt}
                # append row to the dataframe
                df = df._append(new_row, ignore_index=True)

        # self.plot_results(df)

        return df
