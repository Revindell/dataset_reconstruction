from abc import ABC, abstractmethod

import numpy as np
# Strategy interface
from commonroad.scenario.scenario import Scenario


class Strategy(ABC):
    """ Abstract class which creates the interface to the different cleaning algorithms (concrete strategies).
    """

    @staticmethod
    def create_state_list(obs: Scenario.dynamic_obstacles, start_step: int = 0) -> np.ndarray:
        """ Creates a list of chosen state variables.

        :param obs: obstacle need to get the state list
        :param start_step: start step of the state list
        :return: state list
        """

        state_list = []
        for state in obs.prediction.trajectory.state_list[start_step:]:
            new_state = np.array([state.__getattribute__("position")[0],
                                  state.__getattribute__("position")[1],
                                  state.__getattribute__("velocity"),
                                  state.__getattribute__("orientation"),
                                  state.__getattribute__("time_step")])
            state_list.append(new_state)
        state_list = np.array(state_list).squeeze()

        return state_list

    @abstractmethod
    def clean_scenario(self, scenario: Scenario, **kwargs) -> None:
        """ Abstract method to clean the scenario from a problem. The method will be overwritten.

        :param scenario: CommonRoad scenario to be cleaned
        :param kwargs: optional arguments
        :return: None
        """
        pass
