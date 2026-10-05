"""Plot joint tracking and torque for one rollout."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, cost

INK = "#2b2b29"
MUTED = "#8a8980"
GRID = "#e6e5df"
SURFACE = "#fcfcfb"
FLIGHT = "#eef4fc"
COMMANDED = "grey"
LIMIT = "black"
SIDE_COLORS = {"left": "#2a78d6", "right": "#eb6834"}
# Right is drawn thinner, on top: where the sides agree, both stay visible.
SIDE_WIDTHS = {"left": 2.6, "right": 1.3}

LABELS = {"hip_pitch": "hip pitch", "knee": "knee",
          "ankle_pitch": "ankle pitch", "shoulder_pitch": "shoulder pitch"}


def joints(trace, dt, tau_limit, path, title="", seconds=4.0):
    """Write a figure of commanded vs measured angle and torque per joint.

    Env 0 of `trace`, which must come from `Env.rollout(..., joints=True)`.
    Both sides share one command, so it is drawn once. Cut at `seconds`, after
    which only the hold remains. The ball's arrival time is marked.
    """
    q_target = np.degrees(trace["q_target"][0].cpu().numpy())
    q = np.degrees(trace["q"][0].cpu().numpy())
    tau = trace["tau"][0].cpu().numpy()
    steps = min(len(q), int(round(seconds / dt)))
    t = np.arange(steps) * dt

    start, end, length = cost.longest_run(trace["airborne"][:1])
    flight = (start.item() * dt, end.item() * dt) if length.item() else None

    fig, axes = plt.subplots(config.N_JOINTS, 2, figsize=(11, 8.5), sharex=True,
                             facecolor=SURFACE)
    for row, name in enumerate(config.CONTROLLED):
        track, torque = axes[row]
        track.plot(t, q_target[:steps, row], color=COMMANDED, lw=1.3, ls="--",
                   label="commanded", zorder=3)
        for s, side in enumerate(config.SIDES):
            track.plot(t, q[:steps, s, row], color=SIDE_COLORS[side],
                       lw=SIDE_WIDTHS[side], label=side)
            torque.plot(t, tau[:steps, s, row], color=SIDE_COLORS[side],
                        lw=SIDE_WIDTHS[side])
        track.set_ylabel(f"{LABELS[name]}\n(deg)")

        limit = float(tau_limit[row])
        for sign in (1, -1):
            torque.axhline(sign * limit, color=LIMIT, lw=1.2, ls="--")
        torque.text(t[-1], limit, f"±{limit:.0f} N·m limit", color=LIMIT,
                    fontsize=8, ha="right", va="bottom")
        torque.set_ylabel("N·m")

        for ax in (track, torque):
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(GRID)
            ax.yaxis.label.set_color(INK)
            if flight:
                ax.axvspan(*flight, color=FLIGHT, zorder=0)
            ax.axvline(config.ARRIVAL_TIME, color=MUTED, lw=1.0, ls=":")

    axes[0, 0].set_title("tracking", color=INK, loc="left", fontsize=11)
    axes[0, 1].set_title("torque", color=INK, loc="left", fontsize=11)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right",
                      labelcolor=INK, ncols=3)
    if flight:
        axes[0, 1].text(sum(flight) / 2, axes[0, 1].get_ylim()[1], "flight",
                        color=MUTED, fontsize=8, ha="center", va="top")
    axes[0, 1].text(config.ARRIVAL_TIME, axes[0, 1].get_ylim()[0], " ball",
                    color=MUTED, fontsize=8, ha="left", va="bottom")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=INK)

    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
