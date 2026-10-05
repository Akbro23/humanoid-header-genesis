"""Score a batch of rollouts: head the cross at the target, land, don't fall.

    loss = time + ball + impact + hops + strain + heel

`time` separates falls from survivors, `ball` is the task, and the rest order
the survivors by how cleanly they land. `ball` has two branches that meet near
zero, so there is no step at the moment of contact:

    missed:  + gap from forehead to ball, at closest approach before any touch
    headed:  - reward that grows as the ball's miss of the target shrinks

A header is the ball's first robot contact landing on the head; anything else
touching it first is a miss, and its gap stays open.

Each term is bounded by its weight, so the weights can be read against each
other:

    sum(landing penalties) < W_BALL   or never heading beats heading badly
    W_BALL < W_TIME * (1 - t/H)       or a header pays for a fall at time t
"""

import torch

from . import config

W_TIME = 3.0
W_BALL = 1.2   # reward for heading on target, negative
W_MISS = 1.0   # penalty for not heading, positive
W_IMPACT = 0.4
W_HOPS = 0.3
W_STRAIN = 0.2
W_HEEL = 0.15

GAP_REF = 0.5      # metres of closest approach worth the full miss penalty
# Metres of miss at which the header earns half its reward. Soft-saturating
# rather than clipped, so a header metres off target still has a slope.
AIM_REF = 0.5
IMPACT_REF = 15.0  # body weights of landing force above standing
HOPS_REF = 1.50    # airborne seconds outside the main flight
STRAIN_REF = 0.20  # mean fraction of driven actuators at their force limit
HEEL_REF = 0.60    # radians of toe-up sole pitch while grounded

TERM_NAMES = ("time", "ball", "impact", "hops", "strain", "heel")

# How long after contact the ball's outgoing speed is read.
STRIKE_WINDOW_S = 0.05


def longest_run(mask):
    """Locate the longest True run per row, as (start, end exclusive, length).

    The longest rather than the first, so a rebound hop off the landing is not
    mistaken for the jump; and a run rather than the total, so a fallen robot
    lying with its feet in the air does not read as a long flight.
    """
    counted = mask.int().cumsum(1)
    reset = torch.where(mask, torch.zeros_like(counted), counted).cummax(1).values
    run = counted - reset  # length of the run ending at each step
    length, end = run.max(1)
    return end - length + 1, end + 1, length


def first(mask):
    """Index of the first True per row, and whether there is one."""
    return mask.int().argmax(1), mask.any(1)


def aim(ball, after):
    """Return the miss distance and where the ball crossed the goal plane.

    `ball` is (B, n, 3) and `after` (B,) the step it was headed at. The miss is
    the distance in the goal plane from the crossing point to the target. A
    ball that never reaches the plane is scored by its path's closest approach
    to the target point instead, so the miss is always defined.
    """
    n, steps, _ = ball.shape
    idx = torch.arange(steps, device=ball.device)
    rows = torch.arange(n, device=ball.device)
    later = idx > after.unsqueeze(1)
    target = torch.tensor((config.GOAL_X, *config.TARGET), device=ball.device)

    k, crossed = first(later & (ball[..., 0] >= config.GOAL_X))
    k = k.clamp(min=1)
    a, b = ball[rows, k - 1], ball[rows, k]
    f = ((config.GOAL_X - a[:, 0]) / (b[:, 0] - a[:, 0]).clamp(min=1e-9)).clamp(0, 1)
    point = a + f.unsqueeze(1) * (b - a)
    in_plane = (point[:, 1:] - target[1:]).norm(dim=1)

    path = (ball - target).norm(dim=2).masked_fill(~later, float("inf"))
    closest = path.min(1).values
    return torch.where(crossed, in_plane, closest), crossed, point[:, 1:]


def strike(trace, at, dt):
    """Return the ball's speed in and out of contact `at`, and the most a
    perfectly elastic hit could give it: incoming speed plus twice the head's.

    A soft contact is underdamped to make the ball bounce, and pushed too far
    it creates energy. The search would find that, so an outgoing speed above
    the elastic bound is flagged rather than trusted.
    """
    ball, head = trace["ball"], trace["forehead"]
    n, steps, _ = ball.shape
    rows = torch.arange(n, device=ball.device).unsqueeze(1)
    k = at.clamp(min=2, max=steps - 1).unsqueeze(1)
    speed = lambda p, j: (p[rows, j] - p[rows, j - 1]).norm(dim=2) / dt

    window = max(1, round(STRIKE_WINDOW_S / dt))
    after = (k + torch.arange(1, window + 1, device=ball.device)).clamp(max=steps - 1)
    v_in = speed(ball, k - 1)[:, 0]
    v_out = speed(ball, after).amax(1)
    bound = v_in + 2.0 * speed(head, k)[:, 0]
    return v_in, v_out, bound


def measure(trace, dt, settled_com_z, body_weight):
    """Reduce the per-step traces to one raw quantity per member, unweighted."""
    down, airborne = trace["down"], trace["airborne"]
    n, steps = down.shape
    idx = torch.arange(steps, device=down.device)
    rows = torch.arange(n, device=down.device)

    fall_at, fell = first(down)
    fall_at = torch.where(fell, fall_at, steps)
    # Every landing term stops at the fall: a robot on the floor slides, holds
    # odd angles and has its feet up, and `time` already charges for going down.
    before = idx < fall_at.unsqueeze(1)

    start, end, length = longest_run(airborne)
    flew = length > 0
    in_flight = (flew.unsqueeze(1) & (idx >= start.unsqueeze(1))
                 & (idx < end.unsqueeze(1)))
    # Landing force is windowed to after touchdown, so the push-off is not
    # charged as an impact.
    landed = flew & (end < steps)
    peak = (trace["grf"] * (idx >= end.unsqueeze(1))).amax(1)

    touch_at, touched = first(trace["touched"])
    headed = touched & trace["headed"][rows, touch_at]
    # The gap closes to zero at any touch, so it is read only up to the first
    # one: a chest or arm strike keeps the distance its forehead never closed.
    until = ~touched.unsqueeze(1) | (idx <= touch_at.unsqueeze(1))
    gap = trace["gap"].masked_fill(~until, float("inf")).amin(1)
    miss, crossed, point = aim(trace["ball"], touch_at)
    v_in, v_out, bound = strike(trace, touch_at, dt)

    grounded = ~airborne & before
    return {
        "t_fall": fall_at * dt,
        "touched": touched,
        "headed": headed,
        "touch_t": touch_at * dt,
        "gap": gap,
        "miss": miss,
        "crossed": crossed,
        "point": point,
        "v_in": v_in,
        "v_out": v_out,
        # A little slack over the elastic bound for finite differencing.
        "too_fast": headed & (v_out > 1.1 * bound + 0.2),
        "rise": (trace["com_z"].amax(1) - settled_com_z).clamp(min=0.0),
        "impact": torch.where(landed, (peak / body_weight - 1.0).clamp(min=0.0),
                              torch.zeros_like(peak)),
        "hops": (airborne & before & ~in_flight).sum(1) * dt,
        "strain": (trace["saturated"] * before).sum(1) / before.sum(1).clamp(min=1),
        "heel": ((-trace["foot_pitch"]).clamp(min=0.0) * grounded).amax(1),
        "flight": length * dt,
        "gone": trace["base_xy"][rows, (fall_at - 1).clamp(min=0)]
                - trace["base_xy"][:, 0],
    }


def terms(measured, horizon, active=None):
    """Weight and bound each raw quantity into its contribution to the loss.

    `active` selects a subset of TERM_NAMES; None means all of them.
    """
    def unit(quantity, ref):
        return (quantity / ref).clamp(max=1.0)

    ball = torch.where(
        measured["headed"],
        -W_BALL * AIM_REF / (measured["miss"] + AIM_REF),
        W_MISS * unit(measured["gap"], GAP_REF))
    weighted = {
        "time": W_TIME * (1.0 - measured["t_fall"] / horizon),
        "ball": ball,
        "impact": W_IMPACT * unit(measured["impact"], IMPACT_REF),
        "hops": W_HOPS * unit(measured["hops"], HOPS_REF),
        "strain": W_STRAIN * unit(measured["strain"], STRAIN_REF),
        "heel": W_HEEL * unit(measured["heel"], HEEL_REF),
    }
    if active is None:
        return weighted

    unknown = set(active) - set(TERM_NAMES)
    if unknown:
        raise ValueError(f"unknown cost terms {sorted(unknown)}, "
                         f"expected a subset of {TERM_NAMES}")
    return {name: weighted[name] for name in TERM_NAMES if name in active}


def loss(trace, dt, settled_com_z, body_weight, active=None):
    """Score every member of the batch. Returns (losses, measured)."""
    measured = measure(trace, dt, settled_com_z, body_weight)
    horizon = trace["down"].shape[1] * dt
    return sum(terms(measured, horizon, active).values()), measured


def report(measured, horizon, i=0, active=None):
    """Summarise one member of the batch, for the CLI."""
    m = {k: v[i] for k, v in measured.items()}
    t = terms({k: v.unsqueeze(0) for k, v in m.items()}, horizon, active)
    total = sum(v.item() for v in t.values())
    speed = (f"   ball {m['v_in']:.2f} -> {m['v_out']:.2f} m/s"
             + ("  ** faster than an elastic hit allows **" if m["too_fast"] else ""))
    if not m["headed"]:
        ball = f"MISSED by {m['gap']:.3f} m"
    elif m["crossed"]:
        ball = (f"headed at {m['touch_t']:.3f} s, crossed the goal line at "
                f"y {m['point'][0]:+.2f} z {m['point'][1]:.2f} m "
                f"(miss {m['miss']:.3f} m)")
    else:
        ball = (f"headed at {m['touch_t']:.3f} s, never reached the goal line "
                f"(closest {m['miss']:.3f} m)")
    return (f"loss {total:+.4f}   "
            + "   ".join(f"{k} {v.item():+.4f}" for k, v in t.items())
            + f"\n  {ball}{speed if m['headed'] else ''}\n  rise {m['rise']:.3f} m   flight {m['flight']:.3f} s   "
            f"peak {m['impact'] + 1.0:.2f} bw   hops {m['hops']:.3f} s   "
            f"went {m['gone'][0]:+.3f} m   strain {100 * m['strain']:.1f}%   "
            f"heel {torch.rad2deg(m['heel']):.1f} deg   "
            f"t_fall {m['t_fall']:.3f}/{horizon:.2f} s   "
            f"{'stayed up' if m['t_fall'] >= horizon else 'went down'}")
