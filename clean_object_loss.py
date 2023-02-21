import math
from typing import Tuple

from strategy_interface import *
from commonroad.common.solution import VehicleType
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics
from trajectory_optimizer.pm_optimizer import *
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

import numpy as np


class CleanObjectLoss(Strategy):
    """ Class which tackles the problem of object loss in the Argoverse dataset (converted into CommonRoad
    Scenario). A Concrete Strategy in the Strategy Pattern Design.
    The algorithm still has flaws as it cannot cope with static obstacles which are lost (can be solved by applying
    the static obstacles removal algorithm first), it cannot cope with overlapping obstacles (can be solved by applying
    the overlapping obstacles removal algorithm first) and it does show problems, when too many vehicles which are
    closeby are all lost.
    """
    def clean_scenario(self, scenario: Scenario, **kwargs) -> Scenario:
        """ Cleans the dataset from object loss. It overrides the abstract method from the abstract class Strategy.

        :param scenario: scenario before cleansing
        :param kwargs: optional arguments
        :return: scenario after cleansing
        """
        new_scenario = scenario
        # define vehicle
        vehicle = VehicleDynamics.PM(VehicleType.FORD_ESCORT)
        # number of states in the scenario and time step in s
        num_states = 0
        for obstacle in scenario.dynamic_obstacles:
            if num_states < len(obstacle.prediction.trajectory.state_list):
                num_states = len(obstacle.prediction.trajectory.state_list)
        step_size = scenario.dt
        # define error tolerance for velocity differences
        error_tol = 3

        # to count total number of added and deleted states
        num_added_states = 0
        num_deleted_states = 0

        # filter out the static obstacles and collect disappearing/appearing obstacles
        disappearing_obs = []
        for obstacle in scenario.dynamic_obstacles:
            state_list = self.create_state_list(obstacle)
            if len(obstacle.prediction.trajectory.state_list) < num_states and\
                    (state_list[:, 0].ptp() + state_list[:, 1].ptp() > 1):
                disappearing_obs.append(obstacle)

        for obstacle in disappearing_obs:
            states = self.create_state_list(obstacle)
            x = states[:, 0]
            y = states[:, 1]
            time = states[:, 4]
            vel_x, vel_y = self.get_average_vel(x, y, step_size)
            # not all objects have 20 states and more
            try:
                latest_vel_x, latest_vel_y = self.get_latest_vel(x, y, step_size)
            except IndexError as error:
                latest_vel_x = vel_x
                latest_vel_y = vel_y
            for obs in disappearing_obs:
                with open('docs/stats.txt', 'a') as f:
                    if obs.initial_state.time_step > time[-1]:
                        pos = obs.initial_state.position
                        delta_x = pos[0] - x[-1]
                        delta_y = pos[1] - y[-1]
                        delta_t = obs.initial_state.time_step - time[-1]
                        expect_vel_x = delta_x / (delta_t * step_size)
                        expect_vel_y = delta_y / (delta_t * step_size)
                        # difference between average velocity and calculated velocity, if objects are the same
                        vel_x_error = expect_vel_x - vel_x
                        vel_y_error = expect_vel_y - vel_y
                        vel_error = abs(vel_x_error) + abs(vel_y_error)
                        # difference of the position of the new appearing obstacle and the calculated position from
                        # disappearing vehicle
                        calc_x_pos = x[-1] + latest_vel_x * delta_t * step_size
                        calc_y_pos = y[-1] + latest_vel_y * delta_t * step_size
                        pos_x_error = pos[0] - calc_x_pos
                        pos_y_error = pos[1] - calc_y_pos
                        pos_error = abs(pos_x_error) + abs(pos_y_error)
                        states2 = self.create_state_list(obs)
                        x2 = states2[:, 0]
                        y2 = states2[:, 1]
                        velx2, vely2 = self.get_average_vel(x2, y2, step_size)
                        # heuristic to find matches
                        if ((abs(vel_x_error) < error_tol or abs(vel_y_error) < error_tol) and vel_error < 4 and
                                pos_error < 9.5 or pos_error < 4) and self.sign_function(vel_x, vel_y, velx2, vely2):
                            start_state = obstacle.prediction.trajectory.state_list
                            end_state = obs.prediction.trajectory.state_list
                            state_var, n = self.optimize_trajectory(vehicle=vehicle, state_list1=start_state,
                                                                    state_list2=end_state, step_size=step_size)
                            # check if an optimal trajectory was found
                            if state_var is False:
                                continue

                            # print(obstacle.obstacle_id, " and ", obs.obstacle_id, " are merged.")
                            # print(n, " states were removed and ", len(state_var), " states were created.")
                            num_added_states = num_added_states + len(state_var)
                            num_deleted_states = num_deleted_states + n

                            # create new dynamic obstacle
                            new_scenario.remove_obstacle(obstacle)
                            new_scenario.remove_obstacle(obs)
                            dyn_obstacle = self.create_new_dynamic_obstacle(state_var=state_var, n=n, obstacle=obstacle,
                                                                            start_state=start_state, end_state=end_state)
                            new_scenario.add_objects(dyn_obstacle)
                            lines = [str(scenario.scenario_id), "\n", str(obstacle.obstacle_id), "\tadded: ",
                                     str(len(state_var)), "\tdeleted: ", str(n), "\n"]
                            for line in lines:
                                f.write(line)
            f.close()
        return new_scenario

    @staticmethod
    def get_average_vel(x: np.ndarray, y: np.ndarray, step_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculates the average velocity of the object from its state matrix. The last 4 elements are neglected,
        as the vehicle slows down before disappearing.

        :param x: x positions of the dynamic object
        :param y: y positions of the dynamic object
        :param step_size: step size of measurement
        :return: average velocity in x and y direction
        """
        x_vel = []
        y_vel = []
        # toss the last 4 measurements, as the vehicle slows down before disappearing
        for i in range(0, len(x) - 5):
            x_vel.append(x[i + 1] - x[i])
        for i in range(0, len(y) - 5):
            y_vel.append(y[i + 1] - y[i])
        x_vel = np.array(x_vel)
        y_vel = np.array(y_vel)
        x_vel /= step_size
        y_vel /= step_size
        ave_vx = np.mean(x_vel)
        ave_vy = np.mean(y_vel)

        return ave_vx, ave_vy

    @staticmethod
    def get_latest_vel(x: np.ndarray, y: np.ndarray, step_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculates the latest velocity in x and y direction from last 10 measurements.

        :param x: x positions of the dynamic object
        :param y: y positions of the dynamic object
        :param step_size: step size of measurement
        :return: latest velocity in x and y direction
        """
        average_vx_dis = []
        average_vy_dis = []
        for i in range(len(x) - 20, len(x) - 10):
            average_vx_dis.append(x[i + 1] - x[i])
        for i in range(len(y) - 20, len(y) - 10):
            average_vy_dis.append(y[i + 1] - y[i])
        average_vx_dis = np.array(average_vx_dis)
        average_vy_dis = np.array(average_vy_dis)
        average_vx_dis /= step_size
        average_vy_dis /= step_size
        average_vx_dis = np.mean(average_vx_dis)
        average_vy_dis = np.mean(average_vy_dis)

        return average_vx_dis, average_vy_dis

    @staticmethod
    def optimize_trajectory(vehicle: PointMassDynamics, state_list1: np.ndarray, state_list2: np.ndarray,
                            step_size: float) -> Tuple[np.ndarray, int]:
        """ Uses cvxpy module to create optimal trajectory from last state of vehicle which disappears and first state
        of vehicle which appears later on.

        :param vehicle: used vehicle model
        :param state_list1: states of the first vehicle
        :param state_list2: states of the second vehicle
        :param step_size: step size of measurement
        :return: constructed states if possible and position where first state list was cut
        """
        n = 1
        sol_found = False
        state_var = False
        # limit the number of deleted states to 15
        while not sol_found and (n + 10) < len(state_list1) and n < 15:
            # calculate velocity at last position in first obstacle
            vel_x_start = state_list1[-(n-1)].position[0] - state_list1[-n].position[0]
            vel_y_start = state_list1[-(n-1)].position[1] - state_list1[-n].position[1]
            vel_x_start /= step_size
            vel_y_start /= step_size
            # calculate velocity at first position in second obstacle
            vel_x_end = state_list2[1].position[0] - state_list2[0].position[0]
            vel_y_end = state_list2[1].position[1] - state_list2[0].position[1]
            vel_x_end /= step_size
            vel_y_end /= step_size
            # calculate time steps between the states
            num_t = state_list2[0].time_step - state_list1[-n].time_step
            # limit the number of added / estimated states for the trajectory to 50
            if num_t + (n - 1) > 50:
                return False, n
            # generate the state matrices for the optimizer
            start_state = np.array([[state_list1[-n].position[0], state_list1[-n].position[1], vel_x_start,
                                     vel_y_start]])
            end_state = np.array([state_list2[0].position[0], state_list2[0].position[1], vel_x_end, vel_y_end])
            # call optimizer
            model = PMOptimizer(step_size, vehicle)
            state_var, _ = model._reconstruct(start_state.T, end_state.T, num_t)
            # check if solution is found
            if state_var is not False:
                sol_found = True
            else:
                n += 1
        return state_var, n

    @staticmethod
    def create_new_dynamic_obstacle(state_var: np.ndarray, n: int, obstacle: DynamicObstacle,
                                    start_state: np.ndarray, end_state: np.ndarray) -> DynamicObstacle:
        """ Creates new state list and use it to create new dynamic obstacle which is based on the first vehicle.

        :param state_var: states constructed with optimizer
        :param n: position the state list of the first vehicle was cut
        :param obstacle: first vehicle
        :param start_state: state_list of first vehicle
        :param end_state: state_list of second vehicle
        :return: new dynamic obstacle
        """
        start_time_step = start_state[-n].time_step
        new_state_list = start_state[0:-n]
        # create states
        for count, value in enumerate(state_var):
            position = np.array([value[0], value[1]])
            velocity = math.sqrt(pow(value[2], 2) + pow(value[3], 2))
            orientation = math.atan2(value[3], value[2])
            new_state = State(position=position, velocity=velocity, orientation=orientation,
                              time_step=start_time_step + count)
            new_state_list.append(new_state)
        # first state of end_state is already included in optimized states
        new_state_list = new_state_list + end_state[1:]

        # create dynamic object
        obstacle_id = obstacle.obstacle_id
        obstacle_type = ObstacleType.CAR
        obstacle_shape = obstacle.obstacle_shape
        obstacle_initial_state = obstacle.initial_state
        obstacle_trajectory = Trajectory(new_state_list[0].time_step, new_state_list)
        obstacle_prediction = TrajectoryPrediction(obstacle_trajectory, obstacle_shape)
        dynamic_obstacle = DynamicObstacle(obstacle_id=obstacle_id, obstacle_type=obstacle_type,
                                           obstacle_shape=obstacle_shape, initial_state=obstacle_initial_state,
                                           prediction=obstacle_prediction)
        return dynamic_obstacle

    @staticmethod
    def sign_function(velx1: float, vely1: float, velx2: float, vely2: float) -> bool:
        """ Checks whether the (main) direction of movement (of two vehicles) is the same.

        :param velx1: velocity of first vehicle in x direction
        :param vely1: velocity of first vehicle in y direction
        :param velx2: velocity of second vehicle in x direction
        :param vely2: velocity of second vehicle in y direction
        :return: True if vehicles have same (main) direction of movement
        """
        if abs(velx1) > abs(vely1):
            if (velx1 > 0 and velx2 > 0) or (velx1 < 0 and velx2 < 0):
                return True
            else:
                return False
        elif abs(vely1) > abs(velx1):
            if (vely1 > 0 and vely2 > 0) or (vely1 < 0 and vely2 < 0):
                return True
            else:
                return False
        else:
            return False
