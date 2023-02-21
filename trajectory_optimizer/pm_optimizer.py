import sys
from typing import Tuple, Union

try:
    import cvxpy as cp
except ModuleNotFoundError:
    print(
        "Extra dependency cvxpy not found! To install run `pip install "
        "commonroad-drivability-checker[reconstruction]`",
        file=sys.stderr)
    raise

import numpy as np

from trajectory_optimizer.base_optimizer import ReconstructionFailedException, BaseOptimizer
from commonroad_dc.feasibility.vehicle_dynamics import PointMassDynamics


class PMOptimizer(BaseOptimizer):
    def __init__(self, dt, vehicle_dynamics: PointMassDynamics, q_x=None,
                 q_u=None, verbose=False):
        super().__init__(dt, vehicle_dynamics)
        if q_x is not None:
            self.q_x = np.array(q_x)
            assert self.q_x.shape == (4, 4)
        else:
            self.q_x = np.diag([1.0, 1.0, 1.0, 1.0])

        if q_u is not None:
            self.q_u = np.array(q_u)
            assert self.q_u.shape == (2, 2)
        else:
            self.q_u = np.diag([1.0e-5, 1.0e-5])

        self.A = np.eye(4)
        self.A[0, 2] = self.dt
        self.A[1, 3] = self.dt
        self.B = np.array(
                [[0.5 * self.dt * self.dt, 0], [0, 0.5 * self.dt * self.dt],
                    [self.dt, 0], [0, self.dt], ])
        self.verbose = verbose

    def _reconstruct(self, start_state: np.ndarray, end_state: np.ndarray, num_time_steps: int) \
            -> Union[Tuple[np.ndarray, np.ndarray], Tuple[bool, bool]]:
        a_max_squared = self.vehicle_params.longitudinal.a_max ** 2

        # Decision variables
        state_var = cp.Variable(shape=(num_time_steps + 1, 4))
        control_var = cp.Variable(shape=(num_time_steps, 2))

        # initial state
        x_0_var = state_var[0:1].T

        # Cost for initial state deviation
        constraints = [start_state == x_0_var]
        cost = 0
        for k in range(1, state_var.shape[0]):
            # previous state
            x_k_prev_var = state_var[k - 1: k].T
            # Current state variable
            x_k_var = state_var[k: k + 1].T
            # Current control variable for transition from k-1 -> k
            u_k_var = control_var[k - 1: k].T
            # Forward simulation from k-1 to k
            x_k_sim = self.A @ x_k_prev_var + self.B @ u_k_var
            # Cost control magnitude -> control as less as possible
            cost += cp.quad_form(u_k_var, self.q_u)
            # Constraints
            # Continuity constraint (multiple shooting)
            constraints.append(x_k_sim == x_k_var)
            # Friction circle
            constraints.append(cp.sum_squares(u_k_var) <= a_max_squared)
        # last state
        constraints.append(end_state == state_var[-1].T)
        prob = cp.Problem(cp.Minimize(cost), constraints)

        # Solve optimization problem
        try:
            prob.solve(verbose=self.verbose)
            if prob.status != "optimal":
                return False, False
        except:
            return False, False
        return state_var.value, control_var.value

    @staticmethod
    def _state_difference(x_ref: np.ndarray, x_var):
        diff = [x_var[i] - x_ref[i] if not np.isnan(x_ref[i]) else 0. for i in
                range(x_ref.size)]
        return cp.hstack(diff)
