import copy
import math
from collections import Iterable
from typing import Union

from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType, DynamicObstacle, Shape
from commonroad.scenario.trajectory import State, Trajectory
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object, \
    create_collision_checker
import warnings
from strategy_interface import *


class CleanStatic(Strategy):
    """ Class which tackles the problem of static obstacles in the Argoverse dataset (converted into CommonRoad
    Scenario). A Concrete Strategy in the Strategy Pattern Design.
    """

    def clean_scenario(self, scenario: Scenario, roadside_parking_check: bool = False,
                       remove_all: bool = False, **kwargs) -> Scenario:
        """ Cleans the dataset from static obstacles. It overrides the abstract method from the abstract class Strategy.

        :param scenario: scenario before cleansing
        :param roadside_parking_check: if True, the scenario is checked for roadside parking
        :param remove_all: if True, all static obstacles are removed
        :param kwargs: optional arguments
        :return: scenario after cleansing
        """
        # parked vehicles creates warning 'Inequality of Lanelet {self} and the other one {other}'
        warnings.filterwarnings('ignore', '.*inequality.*',)

        new_scenario = scenario
        # create list of stationary obstacles
        stationary_obs_list = self.create_stationary_obstacle_list(new_scenario)

        if remove_all:
            new_scenario.remove_obstacle(stationary_obs_list)
        else:

            for index, obstacle in enumerate(stationary_obs_list):
                new_scenario.remove_obstacle(obstacle)
                state_list = self.create_state_list(obstacle)
                # check if the obstacle is on the intersection
                res_intersection = self.intersection_check_by_surrounding_obstacles(obstacle, stationary_obs_list)
                # res_intersection = self.intersection_check_by_map_information(state_list, scenario)

                # collision check
                # create collision checker from scenario
                cc = create_collision_checker(new_scenario)
                # create collision object for new static object
                obs_co = create_collision_object(obstacle)
                # check if dynamic obstacles collides
                collision_res = cc.collide(obs_co)

                # if the obstacle does not collide with other obstacles
                if not collision_res:
                    if res_intersection:
                        # change the obstacle type to dynamic PARKED_VEHICLE
                        dynamic_parked_vehicle = self.create_dynamic_stationary_vehicle(state_list, new_scenario, obstacle)
                        new_scenario.add_objects(dynamic_parked_vehicle)
                    else:
                        # change the obstacle to static obstacle
                        # create a new static obstacle
                        new_static_obstacle = self.create_static_obstacle(state_list, obstacle.obstacle_id,
                                                                          obstacle.obstacle_shape,
                                                                          static_obstacle_type=ObstacleType.PARKED_VEHICLE)
                        # add the new static obstacle to the scenario
                        new_scenario.add_objects(new_static_obstacle)
                else:
                    pass

            # change vehicles at the road boundary into parked vehicles
            if roadside_parking_check:
                new_scenario = self.parked_at_road_boundary(new_scenario)

                for obstacle in new_scenario.static_obstacles:
                    # collision check
                    new_scenario.remove_obstacle(obstacle)
                    # create collision checker from scenario
                    cc = create_collision_checker(new_scenario)
                    # create collision object for new static object
                    obs_co = create_collision_object(obstacle)
                    # check if dynamic obstacles collides
                    collision_res = cc.collide(obs_co)
                    if not collision_res:
                        new_scenario.add_objects(obstacle)

        return new_scenario

    def create_stationary_obstacle_list(self, scenario: Scenario) -> list:
        """ Creates a list of stationary obstacles from the scenario.

        :param scenario: scenario
        :return: list of stationary obstacles
        """
        stationary_obs_list = []
        for obs in scenario.dynamic_obstacles:
            # get the state list for the obstacle
            state_list = self.create_state_list(obs)

            # check if the obstacle is almost stationary
            if state_list[:, 2].max() < 3 and state_list[:, 1].ptp() < 5 and state_list[:, 0].ptp() < 5:
                stationary_obs_list.append(obs)

        return stationary_obs_list

    @staticmethod
    def create_static_obstacle(state_list: np.array, obstacle_id: int, obstacle_shape: Shape,
                               static_obstacle_type: ObstacleType = ObstacleType.PARKED_VEHICLE) -> StaticObstacle:
        """ Creates a static obstacle from a dynamic obstacle.

        :param state_list: list of states of the dynamic obstacle
        :param obstacle_id: id of the dynamic obstacle
        :param obstacle_shape: shape of the dynamic obstacle
        :param static_obstacle_type: type of the static obstacle
        :return: static obstacle
        """
        # create a new static obstacle
        static_obstacle_id = obstacle_id
        static_obstacle_type = static_obstacle_type
        static_obstacle_shape = obstacle_shape
        static_obstacle_initial_state = State(position=state_list[0, 0:2], orientation=state_list[0, 3],
                                              time_step=0)

        new_static_obstacle = StaticObstacle(static_obstacle_id, static_obstacle_type, static_obstacle_shape,
                                             static_obstacle_initial_state)

        return new_static_obstacle

    def create_dynamic_stationary_vehicle(self, statelist: np.ndarray, scenario: Scenario, obstacle: DynamicObstacle)\
            -> DynamicObstacle:
        """ Creates a dynamic parked vehicle from a dynamic obstacle.

        :param statelist: list of states of the dynamic obstacle
        :param scenario: scenario
        :param obstacle: dynamic obstacle
        :return: dynamic parked vehicle
        """

        # get the time steps for the trajectory
        time_steps = statelist[:, 4]
        initial_time_step = int(time_steps[0])
        # get the initial state of the dynamic obstacle
        initial_position = statelist[0, 0:2]
        initial_orientation_try = self.get_initial_orientation(obstacle, scenario)
        if initial_orientation_try is not None:
            initial_orientation = initial_orientation_try
        else:
            initial_orientation = statelist[0, 3]

        # create states
        new_state_list_converted = []
        for count, value in enumerate(statelist):
            position = initial_position
            velocity = 0
            orientation = initial_orientation
            new_state = State(position=position, velocity=velocity, orientation=orientation,
                              time_step=initial_time_step + count)
            new_state_list_converted.append(new_state)

        # create the trajectory of the obstacle, starting at time step 1
        dynamic_obstacle_trajectory = Trajectory(initial_time_step, new_state_list_converted)

        # create the prediction using the trajectory and the shape of the obstacle
        dynamic_obstacle_shape = obstacle.obstacle_shape
        dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)
        # crate initial state
        dynamic_obstacle_initial_state = State(position=initial_position,
                                               orientation=initial_orientation,
                                               velocity=0, time_step=initial_time_step)
        # generate the dynamic obstacle according to the specification
        dynamic_obstacle_id = scenario.generate_object_id()
        dynamic_obstacle_type = ObstacleType.CAR
        new_dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                               dynamic_obstacle_type,
                                               dynamic_obstacle_shape,
                                               dynamic_obstacle_initial_state,
                                               dynamic_obstacle_prediction)
        return new_dynamic_obstacle

    @staticmethod
    def intersection_check_by_map_information(state_list: np.ndarray, scenario: Scenario) -> Union[None, bool]:
        """ Checks if the obstacle is on an intersection by getting information directly from the map.

        :param state_list: list of states of the dynamic obstacle
        :param scenario: scenario
        :return: True if the obstacle is on an intersection, None otherwise
        """
        # get the id of the lanelet on which the obstacle stays
        lanelet_id = []
        lanelet_id = scenario.lanelet_network.find_lanelet_by_position([state_list[0, 0:2]])
        lanelet_id = flatten(lanelet_id)
        # check if the obstacle on the intersection
        if lanelet_id:
            res_intersection = scenario.lanelet_network.find_intersection_by_id(lanelet_id[0])
        else:
            res_intersection = None

        return res_intersection

    @staticmethod
    def intersection_check_by_surrounding_obstacles(obs: Scenario.dynamic_obstacles, stationary_obs_list: list) -> bool:
        """ Checks if the obstacle is on an intersection by checking if the stationary exhibiting cluster behaviour.

        :param obs: dynamic obstacle
        :param stationary_obs_list: list of stationary obstacles
        :return: True if the obstacle is on an intersection, False otherwise

        """
        surrounding_obstacles_count = 0
        # get the initial position of the being checked obstacle
        initial_position = obs.initial_state.position
        # check if the obstacle has many surrounding stationary obstacles
        # for index, obstacle in enumerate(stationary_obs_list):
        for obstacle in stationary_obs_list:
            # get the initial position of the surrounding obstacle
            surrounding_obstacle_initial_position = obstacle.initial_state.position
            # check if the obstacle is in the surroundings of the being checked obstacle
            if abs(initial_position[0] - surrounding_obstacle_initial_position[0]) < 25 and \
                    abs(initial_position[1] - surrounding_obstacle_initial_position[1]) < 25:
                # count the surrounding obstacles
                surrounding_obstacles_count += 1
        # check if the obstacle is surrounded by many stationary obstacles
        if surrounding_obstacles_count > 3:
            return True
        else:
            return False

    @staticmethod
    def get_initial_orientation(obstacle: Scenario.dynamic_obstacles, scenario: Scenario) -> float:
        """ Calculates the prior orientation of the vehicle according to the lanelet.

        :param obstacle: dynamic obstacle
        :param scenario: scenario
        :return: orientation of the lanelet
        """
        lanelet_orientation = None
        oc = create_collision_object(obstacle)
        for lanelet in scenario.lanelet_network.lanelets:
            lc = create_collision_object(lanelet.polygon)
            if lc.collide(oc):
                diff = lanelet.polygon.vertices[4] - lanelet.polygon.vertices[0]
                orientation = math.atan2(diff[1], diff[0])
                lanelet_orientation = orientation

        return lanelet_orientation

    def parked_at_road_boundary(self, scenario: Scenario) -> Scenario:
        """ Converts dynamic vehicles at the road boundary with nearly no movement into parked vehicles / static
        obstacles.

        :param scenario: Commonroad scenario
        :return: adapted Commonroad scenario
        """
        # lists to collect lanelets / lanelet-collision-objects with no neighboring lanelet
        lanelets = []
        lanelet_c = []

        for lanelet in scenario.lanelet_network.lanelets:
            if lanelet.adj_left is None or lanelet.adj_right is None:
                lc = create_collision_object(lanelet.polygon)
                lanelet_c.append(lc)
                lanelets.append(lanelet)

        # dict of objects to be converted into static ones
        obj = {}
        for obs in scenario.dynamic_obstacles:
            state_list = self.create_state_list(obs)
            if state_list[:, 0].ptp() < 3 and state_list[:, 1].ptp() < 3 and state_list[:, 2].max() < 3:
                oc = create_collision_object(obs)
                obj[obs] = 0
                # count number of collisions with lanelets which are not at the outer bound of the road
                for lanelet in scenario.lanelet_network.lanelets:
                    lc = create_collision_object(lanelet.polygon)
                    if oc.collide(lc) and lanelet not in lanelets:
                        obj[obs] = obj[obs] + 1

        # remove obstacles with more than 1 collision as they are in the middle of the road (or at access/exit ramps)
        copy_obj = copy.copy(obj)
        for key, value in copy_obj.items():
            if value > 1:
                del obj[key]

        # remove obstacles whose position is mainly not on outer lanes of the road
        copy_obj = copy.copy(obj)
        for lane in scenario.lanelet_network.lanelets:
            if lane not in lanelets:
                for obs in copy_obj.keys():
                    state_list = self.create_state_list(obs)
                    x = state_list[:, 0]
                    y = state_list[:, 1]
                    points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
                    if all(lane.contains_points(points)) and obs in obj.keys():
                        del obj[obs]

        # there are some inconsistencies in the road boundary; remove vehicles which are at access/exit ramps or
        # between two lanelets which were classified as outer lanes of the road
        for obs in obj.keys():
            oc = create_collision_object(obs)
            for lanelet in lanelet_c:
                if oc.collide(lanelet):
                    obj[obs] = obj[obs] + 1
        # vehicles can be on the border of two connecting lanelets
        copy_obj = copy.copy(obj)
        for key, value in copy_obj.items():
            if value > 2:
                del obj[key]

        new_scenario = scenario

        for obs in obj.keys():
            static_obstacle_id = obs.obstacle_id
            static_obstacle_type = ObstacleType.PARKED_VEHICLE
            static_obstacle_shape = obs.obstacle_shape
            static_obstacle_initial_state = obs.initial_state
            static_obstacle = StaticObstacle(obstacle_id=static_obstacle_id, obstacle_type=static_obstacle_type,
                                             obstacle_shape=static_obstacle_shape,
                                             initial_state=static_obstacle_initial_state)
            new_scenario.remove_obstacle(obs)
            new_scenario.add_objects(static_obstacle)

        return new_scenario


def flatten(t: list) -> list:
    """ Flattens a list of lists.

    :param t: list of lists
    :return: flattened list
    """
    if isinstance(t, Iterable):
        return [x for sub in t for x in flatten(sub)]
    else:
        return [t]
