from abc import ABCMeta, abstractmethod
from typing import Union, Dict, Tuple

import numpy as np
from commonroad.scenario.trajectory import Trajectory, State


class ReconstructionFailedException(Exception):
    pass


class BaseOptimizer(metaclass=ABCMeta):
    def __init__(self, dt, vehicle_dynamics):
        self.vehicle_dynamics = vehicle_dynamics
        self.dt = dt

    @property
    def vehicle_params(self):
        return self.vehicle_dynamics.parameters

    def reconstruct(
        self, trajectory: Union[Trajectory, np.ndarray, Dict[int, State]]
    ) -> Union[Tuple[Trajectory, Trajectory], Tuple[np.ndarray, np.ndarray]]:
        """
        Reconstruct the trajectory and corresponding inputs

        :param trajectory: Reference trajectory which may either be a Trajectory
        object, a numpy array of state vectors, or a dictionary mapping from
        time step to State objects (can be used if states are missing).
        :return: optimized trajectory as numpy array if reference trajectory is
        given as numpy array otherwise as Trajectory; input trajectory as
        numpy array if reference trajectory is given as numpy array otherwise as
        Trajectory
        """

        if isinstance(trajectory, Trajectory):
            np_trajectory = np.array(
                [
                    self.vehicle_dynamics.state_to_array(s, fill_missing=np.nan)[0]
                    for s in trajectory.state_list
                ]
            )
        elif isinstance(trajectory, dict):
            initial_time_step = min(trajectory.keys())
            final_time_step = max(trajectory.keys())
            np_trajectory = []
            for i in range(initial_time_step, final_time_step + 1):
                state = trajectory.get(i)
                if state is None:
                    state = State(timestep=i)
                np_trajectory.append(state)
            np_trajectory = np.stack(np_trajectory)
        else:
            np_trajectory = trajectory

        res = self._reconstruct(np_trajectory)

        recons_traj, inputs = res
        if isinstance(trajectory, Trajectory):
            recons_traj = Trajectory(
                trajectory.initial_time_step,
                [
                    self.vehicle_dynamics.array_to_state(s, i)
                    for i, s in enumerate(recons_traj, trajectory.initial_time_step)
                ],
            )
            inputs = Trajectory(
                trajectory.initial_time_step,
                [
                    self.vehicle_dynamics.array_to_input(u, i)
                    for i, u in enumerate(inputs, trajectory.initial_time_step)
                ],
            )
        return recons_traj, inputs

    @abstractmethod
    def _reconstruct(self, np_trajectory) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the trajectory and corresponding inputs

        :param np_trajectory: Reference trajectory as numpy array
        :return: optimized trajectory as numpy array, input trajectory as
        numpy array
        """
        pass
