"""Solve the cross backward: from how the ball should arrive to how it is kicked.

The ball arrives at a given point, time and velocity. Its flight before that is
a drag-free parabola, so running it backward until it reaches the ground gives
the kick: where it rests, when it is struck, and how fast.
"""

import math

import numpy as np

from . import config

GRAVITY = 9.81


def arrival_velocity(speed=config.ARRIVAL_SPEED,
                     descent_deg=config.ARRIVAL_DESCENT_DEG):
    """The ball's velocity on arrival: toward the robot (-x), descending."""
    a = math.radians(descent_deg)
    return np.array([-speed * math.cos(a), 0.0, -speed * math.sin(a)])


def launch(arrival_point, dt, arrival_time=config.ARRIVAL_TIME,
           velocity=None, radius=config.BALL_RADIUS):
    """Return (kick step, kick position, kick velocity) for a cross that
    passes `arrival_point` at `arrival_time` with `velocity`.

    The kick position rests on the ground, so the flight time `T` solves
    z_kick = radius for the backward parabola:

        z(T) = P_z - v_z T - g T^2 / 2 = radius

    `T` is then rounded to whole steps, which leaves the kick a little off the
    ground. The simulator integrates velocity before position, which drops a
    falling body by g dt T / 2 over the flight; half a step of gravity on the
    kick velocity cancels that.
    """
    p = np.asarray(arrival_point, dtype=float)
    v = arrival_velocity() if velocity is None else np.asarray(velocity, float)
    height = p[2] - radius
    if height <= 0:
        raise ValueError(f"arrival point {p} is below the ball's radius")

    flight = (-v[2] + math.sqrt(v[2] ** 2 + 2 * GRAVITY * height)) / GRAVITY
    steps = round(flight / dt)
    kick_step = round(arrival_time / dt) - steps
    if kick_step < 0:
        raise ValueError(f"the cross needs {flight:.3f} s of flight but arrives "
                         f"at {arrival_time:.3f} s; arrive later")

    flight = steps * dt
    g = np.array([0.0, 0.0, -GRAVITY])
    kick_velocity = v - g * flight
    kick_position = p - kick_velocity * flight - 0.5 * g * flight ** 2
    kick_velocity[2] += 0.5 * GRAVITY * dt
    return kick_step, kick_position, kick_velocity
