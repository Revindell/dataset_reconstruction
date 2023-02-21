import math

from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle
from commonroad.scenario.trajectory import State, Trajectory
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object

from strategy_interface import *


class CleanRotating(Strategy):
    """ Class which tackles the problem of rotating obstacles in the Argoverse dataset (converted into CommonRoad
    Scenario). A Concrete Strategy in the Strategy Pattern Design.
    """

    def clean_scenario(self, scenario: Scenario, **kwargs) -> Scenario:
        """ Cleans the dataset from rotating obstacles. It overrides the abstract method from the abstract class
        Strategy.

        :param scenario: scenario before cleansing
        :param kwargs: optional arguments
        :return: scenario after cleansing
        """
        new_scenario = scenario
        # calculate the orientation of the road lanelet for vehicle orientation initialization
        dict_obstacles = self.get_initial_orientation(new_scenario)
        for index, (key, value) in enumerate(dict_obstacles.items()):

            obs = key

            # compare the orientation of the road lanelet with the orientation of the obstacle
            # get the state list for the obstacle
            state_list = self.create_state_list(obs)
            lanelet_orientation = value
            if math.fabs(state_list[0, 3] - lanelet_orientation) > (math.pi / 12):
                state_list[0, 3] = lanelet_orientation
                new_scenario.remove_obstacle(obs)
                # create a new dynamic obstacle
                new_dynamic_obstacle = self.create_dynamic_stationary_vehicle(state_list, obs)
                # add the new dynamic obstacle to the scenario
                new_scenario.add_objects(new_dynamic_obstacle)

        return new_scenario

    def get_initial_orientation(self, scenario: Scenario):
        """ Calculates the prior orientation of the vehicle according to the lanelet.

        :param scenario: scenario the algorithm is applied on
        :return: dictionary of obstacles with their prior orientation
        """
        dict_obstacles = {}
        for index, obstacle in enumerate(scenario.dynamic_obstacles):
            states = self.create_state_list(obstacle)
            if states[:, 0].ptp() < 3 and states[:, 1].ptp() < 3 and states[:, 2].max() < 3:
                # get the first two states of the vehicle
                oc = create_collision_object(obstacle)
                for lanelet in scenario.lanelet_network.lanelets:
                    lc = create_collision_object(lanelet.polygon)
                    if lc.collide(oc):
                        diff = lanelet.polygon.vertices[4] - lanelet.polygon.vertices[0]
                        orientation = math.atan2(diff[1], diff[0])
                        dict_obstacles[obstacle] = orientation
        return dict_obstacles

    @staticmethod
    def create_dynamic_stationary_vehicle(statelist: np.ndarray, obstacle: DynamicObstacle) -> DynamicObstacle:
        """ Creates a dynamic parked vehicle from a dynamic obstacle.

        :param statelist: list of states of the dynamic obstacle
        :param obstacle: dynamic obstacle
        :return: dynamic parked vehicle
        """

        # get the time steps for the trajectory
        time_steps = statelist[:, 4]
        initial_time_step = int(time_steps[0])
        # get the initial state of the dynamic obstacle
        initial_position = statelist[0, 0:2]
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
        dynamic_obstacle_id = obstacle.obstacle_id  # scenario.generate_object_id()
        dynamic_obstacle_type = ObstacleType.CAR
        new_dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                               dynamic_obstacle_type,
                                               dynamic_obstacle_shape,
                                               dynamic_obstacle_initial_state,
                                               dynamic_obstacle_prediction)
        return new_dynamic_obstacle
