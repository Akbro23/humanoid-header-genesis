"""Run a whole CMA-ES generation as one batched Genesis scene.

The population is the batch dimension: `n_envs` copies of the H1-2 step together
on the GPU, each tracking its own pose schedule under the same joint PD, each
with its own ball. Every ball is kicked at the same moment with the same
velocity; only what the robots do to them differs.
"""

import math

import numpy as np
import torch
import genesis as gs

from . import config, cross, traj

_initialized = False


def rotate(quat, v):
    """Rotate vectors `v` (..., 3) by unit quaternions `quat` (..., 4), (w, x, y, z)."""
    w, xyz = quat[..., :1], quat[..., 1:]
    xyz, v = torch.broadcast_tensors(xyz, v)
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


class Env:
    """A batched scene, settled and ready to roll out parameter vectors."""

    def __init__(self, n_envs, dt=config.DT, horizon=config.HORIZON_S,
                 camera=False, ball_offset=config.BALL_OFFSET,
                 video_speed=config.VIDEO_SPEED):
        global _initialized
        if not _initialized:
            gs.init(backend=gs.gpu, logging_level="warning")
            _initialized = True

        self.n_envs = n_envs
        self.dt = dt
        self.n_steps = int(round(horizon / dt))
        self.ball_offset = np.array(ball_offset, dtype=float)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=1),
            vis_options=gs.options.VisOptions(
                ambient_light=config.AMBIENT_LIGHT,
                lights=[dict(light) for light in config.LIGHTS],
            ),
            # Recordings are paced by this: 0.5 films twice the frames per
            # simulated second, which plays back at half speed.
            viewer_options=gs.options.ViewerOptions(realtime_factor=video_speed),
            show_viewer=False,
        )
        # The floor comes from Genesis, not the model file: the model's own
        # floor lost its checker texture on import and rendered flat grey.
        self.floor = self.scene.add_entity(gs.morphs.Plane(tile_size=(0.5, 0.5)))
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=str(config.MJCF)))
        volume = 4.0 / 3.0 * math.pi * config.BALL_RADIUS ** 3
        # Parked out of the way until the kick spot is known, after the settle.
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=config.BALL_RADIUS,
                             pos=(config.GOAL_X, 0.0, config.BALL_RADIUS)),
            material=gs.materials.Rigid(rho=config.BALL_MASS / volume,
                                        friction=config.BALL_FRICTION),
            surface=gs.surfaces.Default(color=config.BALL_COLOR),
        )
        self._add_goal()
        # Each camera films env 0 to its own video: the side view and the goal.
        self.cams = {
            name: self.scene.add_camera(
                res=config.RENDER_RES, pos=pos, lookat=lookat,
                fov=config.CAMERA_FOV, GUI=False, env_idx=0)
            for name, pos, lookat in (
                ("run", config.CAMERA_POS, config.CAMERA_LOOKAT),
                ("goal", config.GOAL_CAMERA_POS, config.GOAL_CAMERA_LOOKAT))
        } if camera else {}

        self.scene.build(n_envs=n_envs)
        self.device = self.robot.get_dofs_position().device

        self._index()
        self._settle()
        self._aim_cross()

    # --- setup ------------------------------------------------------------
    def _box(self, lower, upper, color):
        """A fixed box drawn between two corners, with no collision."""
        self.scene.add_entity(
            gs.morphs.Box(lower=lower, upper=upper, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=color),
        )

    def _add_goal(self):
        """Posts and crossbar, and the target outlined in the goal plane."""
        x, w, h, t = config.GOAL_X, config.GOAL_WIDTH / 2, config.GOAL_HEIGHT, config.GOAL_POST
        for y in (-w - t, w):
            self._box((x, y, 0.0), (x + t, y + t, h + t), config.GOAL_COLOR)
        self._box((x, -w - t, h), (x + t, w + t, h + t), config.GOAL_COLOR)

        (ty, tz), s, line = config.TARGET, config.TARGET_SIZE / 2, config.TARGET_LINE
        # Drawn a hair in front of the plane, so the crossing it marks is visible.
        x0, x1 = x - line, x
        for z in (tz - s - line, tz + s):
            self._box((x0, ty - s - line, z), (x1, ty + s + line, z + line),
                      config.TARGET_COLOR)
        for y in (ty - s - line, ty + s):
            self._box((x0, y, tz - s), (x1, y + line, tz + s), config.TARGET_COLOR)

    def _index(self):
        """Resolve joint and link names to the indices the rollout uses."""
        motors = [j for j in self.robot.joints if j.n_dofs == 1]
        self.motor_dofs = [j.dofs_idx_local[0] for j in motors]
        order = {j.name: i for i, j in enumerate(motors)}

        self.ctrl_idx = torch.tensor(
            [[order[f"{side}_{base}_joint"] for base in config.CONTROLLED]
             for side in config.SIDES], device=self.device)

        nominal = torch.zeros(len(motors), device=self.device)
        for name, value in config.NOMINAL_POSE.items():
            nominal[order[name]] = value
        self.nominal = nominal

        # The H1-2 description ships neither, and KP below cannot be integrated
        # stably at zero. Set on the actuated dofs only, so the floating base
        # keeps its true zero.
        n = len(motors)
        self.robot.set_dofs_armature(
            torch.full((n,), config.ARMATURE, device=self.device), self.motor_dofs)
        self.robot.set_dofs_damping(
            torch.full((n,), config.DAMPING, device=self.device), self.motor_dofs)

        _, upper = self.robot.get_dofs_force_range(self.motor_dofs)
        self.tau_limit = upper

        lower, upper = self.robot.get_dofs_limit(self.motor_dofs)
        self.joint_limits = {
            base: (float(lower[order[f"left_{base}_joint"]]),
                   float(upper[order[f"left_{base}_joint"]]))
            for base in config.CONTROLLED
        }

        names = [link.name for link in self.robot.links]
        self.foot_rows = [names.index(n) for n in config.FOOT_LINKS]
        self.base_row = names.index(config.BASE_LINK)
        self.head_row = names.index(config.HEAD_LINK)
        # Contacts report global link ids, which is a different numbering from
        # the rows of `get_links_*` whenever the scene holds more than one entity.
        self.foot_links = torch.tensor(
            [self.robot.links[r].idx for r in self.foot_rows], device=self.device)
        self.head_link = self.robot.links[self.head_row].idx
        # Contacts are filtered by geom, because "any force on a non-foot link"
        # also catches the arms swinging into the legs.
        self.floor_geom = self.floor.geoms[0].idx
        self.ball_geom = self.ball.geoms[0].idx

        for geom, dampratio in [(self.ball.geoms[0], config.BALL_DAMPRATIO)] + [
                (g, config.HEAD_DAMPRATIO) for g in self.robot.links[self.head_row].geoms]:
            params = geom.get_sol_params().clone()
            params[1] = dampratio
            geom.set_sol_params(params)

        self.crown = torch.tensor(config.CROWN, device=self.device)
        self.forehead = torch.tensor(config.FOREHEAD, device=self.device)
        self.up = torch.tensor((0.0, 0.0, 1.0), device=self.device)
        self.body_weight = float(self.robot.get_links_mass().sum()) * 9.81

    def _settle(self):
        """PD-hold the nominal pose until the robot comes to rest, and cache it.

        Every rollout starts from this state, so the settle is paid once rather
        than once per generation.
        """
        target = self.nominal.expand(self.n_envs, -1).contiguous()
        self.robot.set_dofs_position(target, self.motor_dofs)

        for _ in range(int(round(config.SETTLE_SECONDS / self.dt))):
            self._apply(target)
            self.scene.step()

        self.robot.zero_all_dofs_velocity()
        self.settled_qpos = self.robot.get_qpos()[0].clone()
        self.settled_pose = self.robot.get_dofs_position(
            self.motor_dofs)[0][self.ctrl_idx[0]].clone()
        self.settled_com_z = float(self._com_z()[0])

        pos, quat = self._pose()
        head_pos, head_quat = pos[0, self.head_row], quat[0, self.head_row]
        self.settled_crown = (head_pos + rotate(head_quat, self.crown)).cpu().numpy()

    def _aim_cross(self):
        """Place the arrival point off the settled crown and solve the kick."""
        self.arrival = self.settled_crown + self.ball_offset
        self.kick_step, kick_pos, kick_vel = cross.launch(self.arrival, self.dt)
        if self.kick_step >= self.n_steps:
            raise ValueError("the kick falls after the horizon")
        self.kick_pos = torch.tensor(kick_pos, dtype=torch.float32,
                                     device=self.device)
        vel = torch.zeros(6, device=self.device)
        vel[:3] = torch.tensor(kick_vel, dtype=torch.float32)
        self.kick_vel = vel
        # Where it rests until the kick: the kick spot, on the ground.
        self.ball_rest = self.kick_pos.clone()
        self.ball_rest[2] = config.BALL_RADIUS

    # --- physics ----------------------------------------------------------
    def _apply(self, target):
        """Drive the joints toward `target`; return the clipped torques and the
        joint angles they were computed from."""
        q = self.robot.get_dofs_position(self.motor_dofs)
        v = self.robot.get_dofs_velocity(self.motor_dofs)
        tau = config.KP * (target - q) - config.KD * v
        tau = tau.clamp(-self.tau_limit, self.tau_limit)
        self.robot.control_dofs_force(tau, self.motor_dofs)
        return tau, q

    def _pose(self):
        """Fetch link positions and orientations once, for everything that reads
        them. Per-step overhead is what sets the step rate, not the solver."""
        return self.robot.get_links_pos(), self.robot.get_links_quat()

    def _com_z(self):
        """Return the whole-body centre-of-mass height, (B,)."""
        com = self.robot.get_links_pos([self.base_row], ref=gs.link_ref_frame.root_COM)
        return com[:, 0, 2]

    def _contact(self, head_pos, head_quat):
        """Return this step's contact flags and forces, each (B,).

        `grf`, `down` and `airborne` come from the floor contacts: anything but
        a foot on the floor is the fall test. `touched` is any robot contact
        with the ball, and `headed` one on the head link above the neck, judged
        in the head link's frame.
        """
        c = self.robot.get_contacts(is_padded=True)
        valid = c["valid_mask"]

        floor_is_a = c["geom_a"] == self.floor_geom
        on_floor = valid & (floor_is_a | (c["geom_b"] == self.floor_geom))
        link = torch.where(floor_is_a, c["link_b"], c["link_a"])
        # `force_a` acts on geom A, so the robot's side is whichever is not the floor.
        force = torch.where(floor_is_a.unsqueeze(-1), c["force_b"], c["force_a"])
        is_foot = on_floor & (link.unsqueeze(-1) == self.foot_links).any(-1)
        grf = (force[..., 2].abs() * is_foot).sum(1)

        ball_is_a = c["geom_a"] == self.ball_geom
        on_ball = valid & (ball_is_a | (c["geom_b"] == self.ball_geom))
        other = torch.where(ball_is_a, c["link_b"], c["link_a"])
        # Height in the head link's frame: the offset along its own up axis.
        up = rotate(head_quat, self.up).unsqueeze(1)
        height = ((c["position"] - head_pos.unsqueeze(1)) * up).sum(-1)
        on_head = on_ball & (other == self.head_link) & (height > config.NECK_Z)

        return {"grf": grf, "down": (on_floor & ~is_foot).any(1),
                "airborne": ~on_floor.any(1), "touched": on_ball.any(1),
                "headed": on_head.any(1)}

    def _foot_pitch(self, quat):
        """Return the mean sole pitch in radians, (B,); negative is toe up."""
        w, x, y, z = quat[:, self.foot_rows].unbind(-1)
        pitch = torch.atan2(2 * (w * y - x * z), 1 - 2 * (y * y + z * z))
        return pitch.mean(1)

    def reset(self):
        """Put every env back at the settled standing state, ball at rest."""
        self.robot.set_qpos(self.settled_qpos.expand(self.n_envs, -1))
        self.robot.zero_all_dofs_velocity()
        self.ball.set_pos(self.ball_rest.expand(self.n_envs, -1))
        self.ball.zero_all_dofs_velocity()

    def _kick(self):
        """Strike every env's ball from the kick spot."""
        self.ball.set_pos(self.kick_pos.expand(self.n_envs, -1))
        self.ball.set_dofs_velocity(self.kick_vel.expand(self.n_envs, -1))

    # --- rollout ----------------------------------------------------------
    def rollout(self, xs, record=None, joints=False):
        """Simulate one raw parameter vector per env, returning the traces.

        `record` is a folder to film env 0 into, one `<camera>.mp4` per camera,
        or None to skip rendering. The cameras render themselves as the scene
        steps. `joints`
        adds the controlled joints' commanded angle, (B, n_steps, 4), and their
        measured angle and torque per side, (B, n_steps, 2, 4), for plotting.

        `xs` is (B, 27) in raw units, with B at most `n_envs`. A short population
        is padded out and sliced off again.
        """
        xs = np.atleast_2d(xs)
        wanted = len(xs)
        if wanted > self.n_envs:
            raise ValueError(f"{wanted} members into {self.n_envs} envs; "
                             f"chunk the population or build a larger scene")
        if wanted < self.n_envs:
            xs = np.vstack([xs, np.repeat(xs[-1:], self.n_envs - wanted, axis=0)])

        q_traj = traj.target_trajectory(
            xs, self.settled_pose.cpu().numpy(), self.n_steps, self.dt)
        q_traj = torch.as_tensor(q_traj, device=self.device)

        self.reset()
        target = self.nominal.expand(self.n_envs, -1).contiguous()
        limit = self.tau_limit[self.ctrl_idx.flatten()] * 0.999

        shape = (self.n_envs, self.n_steps)
        zeros = lambda *extra, dtype=None: torch.zeros(
            shape + extra, dtype=dtype, device=self.device)
        trace = {
            "down": zeros(dtype=torch.bool),
            "airborne": zeros(dtype=torch.bool),
            "touched": zeros(dtype=torch.bool),
            "headed": zeros(dtype=torch.bool),
            "grf": zeros(),
            "com_z": zeros(),
            "foot_pitch": zeros(),
            "saturated": zeros(),
            "gap": zeros(),
            "base_xy": zeros(2),
            "ball": zeros(3),
            "forehead": zeros(3),
        }
        if joints:
            trace["q_target"] = zeros(config.N_JOINTS)
            trace["q"] = zeros(len(config.SIDES), config.N_JOINTS)
            trace["tau"] = zeros(len(config.SIDES), config.N_JOINTS)
        filming = record is not None and self.cams
        if filming:
            for name, cam in self.cams.items():
                cam.start_recording(save_to_filename=str(record / f"{name}.mp4"),
                                    fps=config.RENDER_FPS)

        for i in range(self.n_steps):
            if i == self.kick_step:
                self._kick()
            # One target per controlled joint, mirrored onto both sides, which
            # is what keeps the motion sagittal.
            target[:, self.ctrl_idx[0]] = q_traj[:, i]
            target[:, self.ctrl_idx[1]] = q_traj[:, i]
            tau, q = self._apply(target)
            if joints:
                trace["q_target"][:, i] = q_traj[:, i]
                trace["q"][:, i] = q[:, self.ctrl_idx]
                trace["tau"][:, i] = tau[:, self.ctrl_idx]
            self.scene.step()

            pos, quat = self._pose()
            head_pos, head_quat = pos[:, self.head_row], quat[:, self.head_row]
            for key, value in self._contact(head_pos, head_quat).items():
                trace[key][:, i] = value
            ball = self.ball.get_pos()
            forehead = head_pos + rotate(head_quat, self.forehead)
            trace["gap"][:, i] = ((forehead - ball).norm(dim=1)
                                  - config.BALL_RADIUS).clamp(min=0.0)
            trace["ball"][:, i] = ball
            trace["forehead"][:, i] = forehead
            trace["com_z"][:, i] = self._com_z()
            trace["foot_pitch"][:, i] = self._foot_pitch(quat)
            # The fraction of the eight driven actuators that are clipped, not
            # whether any is: one pinned ankle should not score like six.
            trace["saturated"][:, i] = (
                tau[:, self.ctrl_idx.flatten()].abs() >= limit).float().mean(1)
            trace["base_xy"][:, i] = pos[:, self.base_row, :2]

        if filming:
            for cam in self.cams.values():
                cam.stop_recording()
        return {key: value[:wanted] for key, value in trace.items()}
