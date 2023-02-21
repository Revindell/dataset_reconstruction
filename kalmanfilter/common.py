import numpy as np


def get_speed_from_position(positions: np.array, dt: float) -> np.array:
    """
    Retrieves speed from position using differentiation
    """
    assert len(positions.shape) and positions.shape[0] > 1 and positions.shape[1] == 2
    v = np.diff(positions, axis=0, append=np.nan) / dt
    speed = np.linalg.norm(v, axis=-1)
    speed = forward_fill_nan(speed)
    return speed


def get_acc_from_positions(positions: np.array, dt: float) -> np.array:
    """
    Retrieves acceleration from position using differentiation
    """
    speed = get_speed_from_position(positions, dt)
    acc = np.diff(speed, append=speed[-1]) / dt
    return acc


def get_theta_from_position(positions: np.array, dt: float) -> np.array:
    """
    Retrieves orientation from position using differentiation and trigonometrics
    """
    assert len(positions.shape) and positions.shape[0] > 1 and positions.shape[1] == 2
    v = np.diff(positions, axis=0, append=np.nan) / dt
    theta = np.arctan2(v[:, 1], v[:, 0])
    return theta


def forward_fill_nan(array: np.array) -> np.array:
    """
    Fills nan's of one-dimensional numpy array with previous non-nan value
    """
    # code copied from https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in
    # -numpy-array
    mask = np.isnan(array)
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, out=idx)
    array_out = array[idx]
    return array_out


def backward_fill_nan(array: np.array) -> np.array:
    """
    Fills nan's of one-dimensional numpy array with following non-nan value
    """
    # code copied from https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in
    # -numpy-array
    mask = np.isnan(array)
    idx = np.where(~mask, np.arange(len(mask)), len(mask) - 1)
    idx = np.minimum.accumulate(idx[::-1])[::-1]
    out = array[idx]
    return out


def get_kappa_from_position(positions: np.array, dt: float) -> np.array:
    """
    Retrieves curvature from velocity (differentiated from position).
    """
    v = (positions[1:] - positions[:-1]) / dt
    a = (v[1:] - v[:-1]) / dt
    kappa = (v[1:, 0] * a[:, 1] - v[1:, 1] * a[:, 0]) / pow((pow(v[1:, 0], 2) + pow(v[1:, 1], 2)), 1 / 3)
    # handle nan's due to division by zero afterwards by forward and backward filling
    if np.isnan(kappa).all():
        # special case where all values are zero
        kappa_out = np.zeros(kappa.shape)
    elif np.isnan(kappa[0]):
        kappa_out = backward_fill_nan(kappa)
    else:
        kappa_out = forward_fill_nan(kappa)
    return kappa_out
