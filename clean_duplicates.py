from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, \
    create_collision_object

from strategy_interface import *


class CleanDuplicates(Strategy):
    """ Class which tackles the problem of duplicate obstacles in the Argoverse dataset (converted into CommonRoad
    Scenario). A Concrete Strategy in the Strategy Pattern Design.
    """

    def clean_scenario(self, scenario: Scenario, **kwargs) -> Scenario:
        """ Cleans the dataset from duplicate obstacles. It overrides the abstract method from the abstract class Strategy.

        :param scenario: scenario before cleansing
        :param kwargs: optional arguments
        :return: scenario after cleansing
        """
        new_scenario = scenario

        for obs in new_scenario.dynamic_obstacles:
            # remove the being checked obstacle first
            new_scenario.remove_obstacle(obs)
            # create collision checker from scenario
            cc = create_collision_checker(scenario)
            obs_co = create_collision_object(obs)
            # check if dynamic obstacles collides
            res = cc.collide(obs_co)

            # if the dynamic obstacle is colliding with other dynamic obstacles add it to the list
            if not res:
                # remove obstacles from the scenario
                new_scenario.add_objects(obs)

        return new_scenario

    @staticmethod
    def get_scenario_dynamic_obstacles_list(scenario: Scenario) -> list:
        """ Returns the list of dynamic obstacles that have collision in the scenario.

        :param scenario: scenario
        :return: list of dynamic obstacles
        """
        # create a list of dynamic obstacles
        dyn_obstacles_list = list()
        # create collision checker from scenario
        cc = create_collision_checker(scenario)
        for dyn_obst in scenario.dynamic_obstacles:
            obs_co = create_collision_object(dyn_obst)
            # check if dynamic obstacles collides
            res = cc.collide(obs_co)
            # print('Collision between the stationary obstacles and other dynamic obstacles: %s' % res)

            # if the dynamic obstacle is colliding with other dynamic obstacles add it to the list
            if res:
                dyn_obstacles_list.append(dyn_obst)

        return dyn_obstacles_list
