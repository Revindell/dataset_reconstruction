from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle, create_road_polygons
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object, \
    create_collision_checker
from commonroad.scenario.obstacle import Obstacle
from commonroad_dc.pycrcc import Shape

from strategy_interface import *


class CleanOffroad(Strategy):
    """ Class which tackles the problem of off-road detections in the Argoverse dataset (converted into CommonRoad
    Scenario). A Concrete Strategy in the Strategy Pattern Design.
    """

    def clean_scenario(self, scenario: Scenario, use_collision_check: bool = False, **kwargs) -> Scenario:
        """ Cleans the dataset from off-road detections. It overrides the abstract method from the abstract class
        Strategy.

        :param scenario: scenario before cleansing
        :param use_collision_check: if True use collision check additionally,
            otherwise use only contains_points function of lanelets
        :param kwargs: optional arguments
        :return: scenario after cleansing
        """
        new_scenario = scenario

        if use_collision_check:
            # create the road boundary
            road_boundary_obstacle, road_boundary_sg_rectangles = \
                create_road_boundary_obstacle(new_scenario, method='aligned_triangulation')
            # create the road polygons
            road_boundary_sg_polygons = create_road_polygons(new_scenario, method='lane_polygons', buffer=1,
                                                             resample=1, triangulate=False)

        # draw road boundary or road polygons if needed
        # draw_road_boundary(scenario, road_boundary_sg_rectangles)
        # draw_road_polygon(scenario, road_boundary_sg_polygons)

        for obs in new_scenario.obstacles:
            if use_collision_check:
                # remove the obstacle from the scenario first, so that it can be later added again
                new_scenario.remove_obstacle(obs)

                # do the collision check
                res_boundary = self.road_boundary_collision_check(scenario, obs, road_boundary_sg_rectangles)
                res_lane = self.road_polygon_collision_check(scenario, obs, road_boundary_sg_polygons)
                # add the obstacle back to the scenario
                if not res_boundary and res_lane:
                    new_scenario.add_objects(obs)

            # check if the obstacle is on the lanelet
            contained = self.on_lanelet_check(scenario, obs)
            # if the obstacle is not on the lanelet, remove it
            if contained:
                pass
            else:
                new_scenario.remove_obstacle(obs)

        return new_scenario

    @staticmethod
    def road_boundary_collision_check(scenario: Scenario, obs: Obstacle, road_boundary_sg_rectangles: Obstacle) -> bool:
        """ Checks if the obstacle is on the road boundary.

        :param scenario: scenario that has been worked on
        :param obs: obstacle need to get the state list
        :param road_boundary_sg_rectangles: road boundary
        :return: True if the obstacle is on the road boundary, False otherwise
        """

        # create collision checker from scenario
        ccb = create_collision_checker(scenario)
        # add road boundary to collision checker
        ccb.add_collision_object(road_boundary_sg_rectangles)
        res_boundary = ccb.collide(create_collision_object(obs))
        # print('Collision between the dynamic obstacles and the road boundary: %s' % res_boundary)

        return res_boundary

    @staticmethod
    def road_polygon_collision_check(scenario: Scenario, obs: Obstacle, road_boundary_sg_polygons: Shape) -> bool:
        """
        Checks if the obstacle is on the road polygons.

        :param scenario: scenario that has been worked on
        :param obs: obstacle need to get the state list
        :param road_boundary_sg_polygons: road polygons
        :return: True if the obstacle is on the road polygons, False otherwise
        """
        # create collision checker from scenario
        ccp = create_collision_checker(scenario)
        # add road polygons to collision checker
        ccp.add_collision_object(road_boundary_sg_polygons)
        res_lane = ccp.collide(create_collision_object(obs))
        # print('Collision between the dynamic obstacles and the road lane: %s' % res_lane)

        return res_lane

    def on_lanelet_check(self, scenario: Scenario, obs: Obstacle) -> bool:
        """
        Checks if the obstacle is on the lanelet.

        :param scenario: scenario that has been worked on
        :param obs: obstacle need to be checked if it is on the lanelet
        :return: True if the obstacle is on the lanelet, False otherwise
        """
        contained = False
        for lanelet in scenario.lanelet_network.lanelets:
            state_list = self.create_state_list(obs)
            x = state_list[:, 0]
            y = state_list[:, 1]
            points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
            if any(lanelet.contains_points(points)):
                contained = True
                break

        return contained

    @staticmethod
    def draw_road_boundary(scenario: Scenario, road_boundary_sg_rectangles: Shape) -> None:
        """ Draw the road boundary.

        :param scenario: scenario that has been worked on
        :param road_boundary_sg_rectangles: road boundary
        :return: None
        """

        # draw the road boundary
        rnd = MPRenderer(figsize=(25, 10))
        road_boundary_sg_rectangles.draw(rnd)
        scenario.lanelet_network.draw(rnd)
        rnd.render()

    @staticmethod
    def draw_road_polygon(scenario: Scenario, road_boundary_sg_polygons: Shape) -> None:
        """ Draw the road polygons.

        :param scenario: scenario that has been worked on
        :param road_boundary_sg_polygons: road polygons
        :return: None
        """

        # draw the road polygons
        rnd = MPRenderer(figsize=(25, 10))
        road_boundary_sg_polygons.draw(rnd, draw_params={'draw_mesh': False})
        scenario.lanelet_network.draw(rnd)
        rnd.render()
