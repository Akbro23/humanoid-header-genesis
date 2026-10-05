"""Search the unit box with CMA-ES, evaluating a whole population per rollout.

The search runs in the unit box, not in raw parameters -- radians and seconds
have no common scale. The box is a margin around the start point rather than the
joint limits, which is what stops the cost being minimized by deleting the jump.

Two stages: `time` + `ball` first, to find the
contact geometry and timing, which is the hard part; then the full cost,
restarted around that result at a much smaller step, to fix the landing. Stage
1's result falls, and it sits on a knife edge: at a wider second step the
population drifts onto headers that fall and stays there.
"""

import time
from typing import NamedTuple

import numpy as np
import cma

from . import config, cost, traj


class Stage(NamedTuple):
    name: str
    terms: tuple[str, ...] | None  # None is every term
    sigma: float                   # initial step, as a fraction of the box
    iters: int | None              # fixed generation budget, or None for the rest


STAGES = (
    Stage("ball", ("time", "ball"), 0.25, 10),
    Stage("land", None, 0.03, None),
)


class Budget:
    """Generations and wall-clock shared by every stage; either may be None."""

    def __init__(self, iters=None, minutes=None):
        self.left = iters
        self.deadline = None if minutes is None else time.monotonic() + 60 * minutes

    def spend(self):
        if self.left is not None:
            self.left -= 1

    @property
    def done(self):
        out_of_iters = self.left is not None and self.left <= 0
        out_of_time = self.deadline is not None and time.monotonic() > self.deadline
        return out_of_iters or out_of_time


def evaluate(env, xs, active=None):
    """Score raw parameter vectors, in as many rollouts as `n_envs` requires.

    Returns the losses and the batch's measured quantities, concatenated.
    """
    losses, measured = [], []
    for start in range(0, len(xs), env.n_envs):
        trace = env.rollout(xs[start:start + env.n_envs])
        batch, m = cost.loss(trace, env.dt, env.settled_com_z,
                             env.body_weight, active)
        losses.append(batch.cpu().numpy().astype(float))
        measured.append({k: v.cpu().numpy() for k, v in m.items()})
    return (np.concatenate(losses),
            {k: np.concatenate([m[k] for m in measured]) for k in measured[0]})


def search(env, x0, sigma, budget, popsize, seed=0, active=None, iters=None,
           log=print):
    """Run one CMA-ES around `x0` until `budget` (or `iters`, if given) runs
    out. Returns the best raw parameters and their loss."""
    lo, hi = traj.bounds(env.joint_limits, x0)
    es = cma.CMAEvolutionStrategy(
        traj.normalize(x0, lo, hi), sigma,
        {"popsize": popsize, "bounds": [0, 1], "seed": seed + 1, "verbose": -9})
    horizon = env.n_steps * env.dt
    best_x, best_loss, gen = np.array(x0, dtype=float), np.inf, 0

    while not budget.done and (iters is None or gen < iters):
        us = np.array(es.ask())
        xs = traj.denormalize(us, lo, hi)
        losses, m = evaluate(env, xs, active)
        es.tell(list(us), losses.tolist())
        budget.spend()
        gen += 1

        i = int(np.argmin(losses))
        if losses[i] < best_loss:
            best_x, best_loss = xs[i].copy(), float(losses[i])
        log(f"gen {gen:3d}  best {best_loss:+.4f}  gen {losses[i]:+.4f}  "
            f"mean {losses.mean():+.4f}  "
            f"headed {100 * m['headed'].mean():3.0f}%  "
            f"up {100 * (m['t_fall'] >= horizon).mean():3.0f}%"
            + (f"  too fast {int(m['too_fast'].sum())}" if m["too_fast"].any() else ""))
    return best_x, best_loss


def staged(env, x0, budget, popsize, seed=0, log=print):
    """Run STAGES in order, each starting from the last one's best."""
    x, loss = np.array(x0, dtype=float), np.inf
    for stage in STAGES:
        if budget.done:
            break
        log(f"--- stage {stage.name}: terms "
            f"{','.join(stage.terms or cost.TERM_NAMES)}, sigma {stage.sigma}")
        x, loss = search(env, x, stage.sigma, budget, popsize, seed=seed,
                         active=stage.terms, iters=stage.iters, log=log)
    return x, loss


def probe(env, x0, sigma, popsize, seed=0, log=print):
    """Score one generation sampled around `x0` and summarise how close the
    head gets to the ball: is the cross within reach of this search?"""
    lo, hi = traj.bounds(env.joint_limits, x0)
    es = cma.CMAEvolutionStrategy(
        traj.normalize(x0, lo, hi), sigma,
        {"popsize": popsize, "bounds": [0, 1], "seed": seed + 1, "verbose": -9})
    _, m = evaluate(env, traj.denormalize(np.array(es.ask()), lo, hi))

    gap = np.where(m["headed"], 0.0, m["gap"])
    log(f"probe: {len(gap)} samples at sigma {sigma}")
    log(f"  headed {100 * m['headed'].mean():.1f}%   "
        f"touched by something else first "
        f"{100 * (m['touched'] & ~m['headed']).mean():.1f}%")
    log("  closest forehead gap (m), percentiles 0/10/50/90: "
        + " / ".join(f"{np.percentile(gap, q):.3f}" for q in (0, 10, 50, 90)))
    log(f"  rise (m), max {m['rise'].max():.3f}   median {np.median(m['rise']):.3f}")
