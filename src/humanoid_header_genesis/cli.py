"""Optimize a keyframe header of an incoming cross for the Unitree H1-2 with
CMA-ES in Genesis."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config, cost, env, opt, plot

DEFAULT_ITERS = 60


def parse_args(argv=None):
    """Build the command line. With no arguments, optimize from the config seed."""
    p = argparse.ArgumentParser(
        prog="header", description=__doc__,
        epilog="RUN is a folder in out/, a file in trajectories/, "
               "or a path to an .npy.")
    p.add_argument("--iter", type=int,
                   help=f"CMA-ES generations, over all stages (default: "
                        f"{DEFAULT_ITERS}, or unlimited when --minutes is set)")
    p.add_argument("--minutes", type=float,
                   help="wall-clock budget, checked once per generation; with "
                        "--iter, whichever runs out first")
    p.add_argument("--popsize", type=int, default=512,
                   help="CMA-ES population; split across --envs if larger")
    p.add_argument("--envs", type=int, default=None,
                   help="scene batch size (default: --popsize)")
    p.add_argument("--sigma", type=float, default=None,
                   help="initial CMA-ES step, as a fraction of the search box "
                        "(default: each stage's own; with --no-stages or "
                        "--probe, 0.25)")
    p.add_argument("--seed", type=int, default=0, help="CMA-ES random seed")
    p.add_argument("--no-stages", dest="stages", action="store_false",
                   help="one search on --terms instead of the staged search")
    p.add_argument("--terms", type=lambda s: tuple(s.split(",")),
                   help=f"with --no-stages, the cost terms to score, comma "
                        f"separated (default: all of {','.join(cost.TERM_NAMES)})")
    p.add_argument("--load", metavar="RUN",
                   help="warm-start from a saved run, centring the search box on it")
    p.add_argument("--replay", nargs="?", const="", metavar="RUN",
                   help="score and render a saved run without searching; with no "
                        "argument, the hand-written seed")
    p.add_argument("--probe", action="store_true",
                   help="score one sampled generation and report how close the "
                        "head gets to the ball, then stop")
    p.add_argument("--ball-offset", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   default=config.BALL_OFFSET,
                   help="ball centre at arrival, relative to the settled crown")
    p.add_argument("--dt", type=float, default=config.DT,
                   help="simulation step, seconds")
    p.add_argument("--horizon", type=float, default=config.HORIZON_S,
                   help="episode length, seconds")
    p.add_argument("--name",
                   help="run folder under out/ (default: date and time; for a "
                        "replay, the folder being replayed)")
    p.add_argument("--video-speed", type=float, default=config.VIDEO_SPEED,
                   help="video playback speed; 0.5 is half speed")
    p.add_argument("--no-video", action="store_true", help="skip rendering run.mp4 and goal.mp4")
    return p.parse_args(argv)


def saved(name):
    """Find a saved parameter vector: a run folder, or a path to an .npy.

    Bare names are looked up in out/ and trajectories/ as well, with or without
    the .npy suffix.
    """
    for path in (Path(name), config.OUT_DIR / name, config.TRAJECTORY_DIR / name):
        if path.is_dir():
            return path / "best.npy"
        for file in (path, path.with_name(path.name + ".npy")):
            if file.is_file():
                return file
    raise FileNotFoundError(f"no run folder or .npy called {name!r} (also looked "
                            f"in {config.OUT_DIR} and {config.TRAJECTORY_DIR})")


def run_folder(args, source):
    """Pick the folder this invocation writes into."""
    if args.name:
        return config.OUT_DIR / args.name
    if source is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return config.OUT_DIR / ("seed" if args.replay == "" else stamp)
    # A replayed run writes back into its own folder; a loose .npy gets one.
    return source.parent if source.name == "best.npy" else config.OUT_DIR / source.stem


def main(argv=None):
    """Optimize the header, then save and render the winner."""
    args = parse_args(argv)
    replaying = args.replay is not None
    n_envs = args.envs or (1 if replaying else args.popsize)
    # The search scene has no camera: filming from a large batch is slow, so the
    # winner is replayed in a scene of its own.
    make = lambda n, camera: env.Env(n_envs=n, dt=args.dt, horizon=args.horizon,
                                     camera=camera, ball_offset=args.ball_offset,
                                     video_speed=args.video_speed)
    scene = make(n_envs, replaying and not args.no_video)
    horizon = scene.n_steps * scene.dt
    print(f"envs {n_envs}   steps {scene.n_steps}   "
          f"com z {scene.settled_com_z:.4f} m   weight {scene.body_weight:.1f} N")
    print(f"cross: kicked at {scene.kick_step * scene.dt:.3f} s from "
          f"x {scene.kick_pos[0]:.2f} m, arrives at {config.ARRIVAL_TIME:.3f} s "
          f"at {np.round(scene.arrival, 3)}")

    x0 = (np.load(saved(args.load)) if args.load is not None
          else np.array(config.SEED, dtype=float))
    if args.probe:
        opt.probe(scene, x0, args.sigma or 0.25, args.popsize, seed=args.seed)
        return

    source = saved(args.replay) if args.replay else None
    folder = run_folder(args, source)
    folder.mkdir(parents=True, exist_ok=True)

    if replaying:
        best = np.load(source) if source is not None else x0
    else:
        # Whichever budget is set does the stopping; --minutes alone means the
        # clock decides, so the generation cap only applies when nothing else would.
        iters = args.iter if args.iter is not None else (
            None if args.minutes else DEFAULT_ITERS)
        budget = opt.Budget(iters, args.minutes)
        if args.stages:
            best, loss = opt.staged(scene, x0, budget, args.popsize, seed=args.seed)
        else:
            best, loss = opt.search(scene, x0, args.sigma or 0.25, budget,
                                    args.popsize, seed=args.seed,
                                    active=args.terms)
        print(f"loss {loss:+.4f}")

    np.save(folder / "best.npy", best)
    if not replaying:
        scene = make(1, not args.no_video)
    trace = scene.rollout(best[None], joints=True,
                          record=None if args.no_video else folder)
    active = None if args.stages else args.terms
    _, measured = cost.loss(trace, scene.dt, scene.settled_com_z,
                            scene.body_weight, active)
    print(cost.report(measured, horizon, active=active))
    for name, value in zip(config.PARAM_NAMES, best):
        print(f"  {name:28s} {value:+.4f}")

    plot.joints(trace, scene.dt, scene.tau_limit[scene.ctrl_idx[0]].cpu(),
                folder / "joints.png", title=folder.name)
    print(f"wrote {folder}/")


if __name__ == "__main__":
    main()
