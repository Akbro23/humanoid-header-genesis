"""Build the pose schedule from the parameter vector, for a whole population.

    x = [ pose(5 frames x 4 joints), T(6), T_wait ]

Durations are deltas, so knot times increase by construction. The schedule
holds home for `T_wait`, runs the manoeuvre, and holds the recovered stance
flat to the end of the episode.

Numpy, like the CMA-ES side that feeds it. Each member has its own knot times,
so the interpolation is a loop over the population.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import config

DURATIONS = slice(config.N_POSE_PARAMS, config.N_POSE_PARAMS + config.N_DURATIONS)
WAIT = config.N_PARAMS - 1


def bounds(joint_limits, x0=None):
    """Return the search box: a margin around `x0`, clipped to joint limits.

    Centred on the config seed by default. A warm start moves the centre with
    it; a box left around the seed would mostly sample the seed's failures.
    The wait is the exception: it spans its whole range wherever the centre is.
    """
    margin = np.array(
        [config.FRAME_MARGIN[f] for f in config.FRAMES for _ in config.CONTROLLED]
        + [config.DURATION_MARGIN[s] for s in config.SEGMENTS] + [0.0])
    centre = np.array(config.SEED if x0 is None else x0, dtype=float)
    lo, hi = centre - margin, centre + margin

    pose = slice(0, config.N_POSE_PARAMS)
    joint_lo, joint_hi = np.array(
        [joint_limits[j] for j in config.CONTROLLED]).T
    lo[pose] = np.maximum(lo[pose], np.tile(joint_lo, config.N_FRAMES))
    hi[pose] = np.minimum(hi[pose], np.tile(joint_hi, config.N_FRAMES))
    lo[DURATIONS] = np.maximum(lo[DURATIONS], config.MIN_DURATION)
    lo[WAIT], hi[WAIT] = config.WAIT_RANGE
    return lo, hi


def denormalize(u, lo, hi):
    """Map unit-box vectors to raw parameters, so radians never share a scale
    with seconds."""
    return lo + np.clip(u, 0.0, 1.0) * (hi - lo)


def normalize(x, lo, hi):
    """Map raw parameters back to the unit box."""
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def schedule(x, settled_pose, horizon):
    """Build the (9, 4) knot poses and their (9,) times for one member.

    The settled stance appears four times: at the start, at the end of the
    wait, as the knot the last searched frame ramps back to, and at the
    horizon so it is held flat. A zero wait is nudged open so the knot times
    stay strictly increasing.
    """
    frames = x[:config.N_POSE_PARAMS].reshape(config.N_FRAMES, config.N_JOINTS)
    wait = max(float(x[WAIT]), 1e-3)

    values = np.vstack([settled_pose, settled_pose, frames, settled_pose,
                        settled_pose])
    knots = wait + np.concatenate([[0.0], np.cumsum(x[DURATIONS])])
    times = np.concatenate([[0.0], knots, [max(horizon, knots[-1] + 1e-3)]])
    return values, times


def target_trajectory(xs, settled_pose, n_steps, dt):
    """Interpolate the knots into the per-step PD setpoints, (B, n_steps, 4).

    One commanded angle per controlled joint per step, which the caller mirrors
    onto both legs and both arms. Past the last knot the pose is held.
    """
    xs = np.atleast_2d(xs)
    ts = np.arange(n_steps) * dt
    out = np.empty((len(xs), n_steps, config.N_JOINTS), dtype=np.float32)

    for i, x in enumerate(xs):
        values, times = schedule(x, settled_pose, n_steps * dt)
        out[i] = PchipInterpolator(times, values, axis=0)(
            np.clip(ts, times[0], times[-1]))
    return out
