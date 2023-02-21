from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import Saver
from pykalman.standard import _smooth_pair, _loglikelihoods
from tqdm import tqdm
from kalmanfilter.ekf_base import BaseEKFWithSmoother


def maximization_step(
        transition_matrices: np.ndarray,
        smoothed_state_means: np.ndarray,
        smoothed_state_covariances: np.ndarray,
        pairwise_covariances: np.ndarray,
        kf: BaseEKFWithSmoother,
) -> np.ndarray:
    """ Maximization step of the EM algorithm.

    :param transition_matrices: transition matrices of the applied
    :param smoothed_state_means: smoothed states
    :param smoothed_state_covariances: smoothed covariances
    :param pairwise_covariances: pairwise covariances
    :param kf: used Kalman Filter
    """
    I = np.eye(transition_matrices.shape[1])
    M = np.zeros_like(smoothed_state_covariances[0])
    for k in range(1, transition_matrices.shape[0]):
        sigma_k_prev = smoothed_state_covariances[k - 1]
        sigma_k = smoothed_state_covariances[k]
        pairwise = pairwise_covariances[k]
        A = transition_matrices[k - 1]
        sigma_tilde = np.linalg.cholesky(
            np.bmat([[sigma_k_prev, pairwise.T], [pairwise, sigma_k]])
        )
        gamma = np.bmat([[-A, I]]) @ sigma_tilde
        # m = np.bmat([[-A, I]]) @ np.bmat([[sigma_k_prev, pairwise.T], [pairwise, sigma_k]])
        x_k = smoothed_state_means[k]
        err = x_k - kf.propagate_x(x_k)
        err = np.outer(err, err)
        # Equation (19) in
        # J. E. Stellet, F. Straub, J. Schumacher, W. Branz, and J. M. Zöllner, “Estimating the Process Noise
        # Variance for Vehicle Motion Models,” in 2015 IEEE 18th International Conference on Intelligent
        # Transportation Systems, Sep. 2015, pp. 1512–1519. doi: 10.1109/ITSC.2015.212.
        M_k = gamma @ gamma.T + err
        # M_k = m @ m.T + err
        M += M_k
    return M.T / (transition_matrices.shape[0] - 1)


def em_process_covariance(kf: BaseEKFWithSmoother, measurements: np.ndarray, init_steps=25) \
        -> Tuple[np.ndarray, np.ndarray, int]:
    """ Computes one EM step.

    :param kf: used Kalman Filter
    :param measurements: measurements used for E-step
    :param init_steps: number of steps for initialization of the Kalman Filter
    """
    # init phase of KF to obtain an initial state estimate with covariance
    for measurement in measurements[:init_steps]:
        kf.predict()
        kf.update_from_measurement(measurement[:2])

    # Expectation step
    # Log all products from now
    saver = Saver(kf)
    transition_matrices = []
    observation_matrices = []
    for measurement in measurements[init_steps:]:
        kf.predict()
        observation_matrices.append(kf.calculate_H_jacobian_at(kf.x))
        kf.update_from_measurement(measurement[:2])
        transition_matrices.append(kf.F)
        saver.save()

    transition_matrices = np.array(transition_matrices)
    saver.to_array()
    likelihood = np.sum(
        _loglikelihoods(
            observation_matrices,
            np.zeros(kf.dim_z),
            kf.R,
            saver.xs_prior,
            saver.Ps_prior,
            measurements[init_steps:, :2],
        )
    )
    state_smoothed, state_cov_smoothed, smoother_gains, __ = kf.rts_smoother(
        saver.xs, saver.Ps
    )
    # Equation (22) in https://users.ece.cmu.edu/~byronyu/papers/derive_eks.pdf
    cov_pair_smooth = _smooth_pair(state_cov_smoothed, smoother_gains)
    em_proc_cov = maximization_step(
        transition_matrices, state_smoothed, state_cov_smoothed, cov_pair_smooth, kf
    )
    return em_proc_cov, likelihood, transition_matrices.shape[0] - 1


def em_kalmanfilter(
        states: List[np.ndarray],
        kf: BaseEKFWithSmoother,
        dt: float,
        num_iterations: int = 20,
        num_init_steps: int = 5,
) -> np.ndarray:
    """ Applies EM algorithm on Kalman Filter to estimate the process noise matrix Q.

    :param states: measurements of the scenario
    :param kf: used Kalman Filter
    :param dt: step size of measurement in seconds
    :param num_iterations: number of iterations the EM algorithm should be applied
    :param num_init_steps: number of steps used for initialization of Kalman Filter
    """
    pruned_states = []
    for state in states:
        if len(state) > num_init_steps + 10:
            pruned_states.append(state)

    em_transition_cov = np.eye(kf.dim_x) * dt
    likelihoods = []
    for _ in tqdm(range(num_iterations), desc="EM Iterations"):
        new_em_transition_cov = np.zeros_like(em_transition_cov)
        likelihoods.append(0.0)
        steps = 0
        for measurement in pruned_states:
            # Select some random sequence starting point
            kf.init_x(measurement[0][0], measurement[0][1], measurement[0][2], measurement[0][3])
            kf.Q = em_transition_cov
            em_proc_cov, likelihood, step = em_process_covariance(
                kf, measurement[1:], num_init_steps
            )
            likelihoods[-1] += likelihood
            new_em_transition_cov += em_proc_cov * step
            steps += step

        # ToDo: Line search (Slide 67 https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec13-KalmanSmoother-MAP-ML-EM.pdf)
        em_transition_cov = new_em_transition_cov / steps

    # plt.plot(likelihoods)
    # plt.show()
    # print(em_transition_cov.tolist())
    # print(em_transition_cov.round(3))

    return em_transition_cov
