from kalmanfilter.em_kf import *
from commonroad.scenario.scenario import Scenario


def estimate_process_noise(scenario: Scenario, kf: BaseEKFWithSmoother, num_iterations: int = 20,
                           num_init_steps: int = 15) -> np.ndarray:
    """ Applies the EM algorithm on dynamic obstacles.

    :param scenario: Scenario to use EM algorithm on to estimate process noise
    :param kf: used Kalman Filter
    :param num_iterations: number of iterations of the EM algorithm
    :param num_init_steps: number of steps for initialization of Kalman Filter
    """
    measurements = []
    dt = scenario.dt
    for obstacle in scenario.dynamic_obstacles:
        state_list = create_state_list(obstacle)
        # filter out static obstacles
        if state_list[:, 0].ptp() > 3 and state_list[:, 1].ptp() > 3 and state_list[:, 2].max() > 3:
            measurements.append(state_list)
    return em_kalmanfilter(measurements, kf, dt, num_iterations, num_init_steps)


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
                              state.__getattribute__("orientation")])
        state_list.append(new_state)
    state_list = np.array(state_list).squeeze()

    return state_list
