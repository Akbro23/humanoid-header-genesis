"""Task constants: the robot, the plant, the cross, the goal and the search box."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MJCF = REPO_ROOT / "models" / "h1_2" / "h1_2_handless.xml"
OUT_DIR = REPO_ROOT / "out"
TRAJECTORY_DIR = REPO_ROOT / "trajectories"

# --- joints ---------------------------------------------------------------
SIDES = ("left", "right")
# The joints the trajectory drives, mirrored left/right to keep the motion
# sagittal. Every other actuated joint holds its nominal target.
CONTROLLED = ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch")

# Standing pose; every actuated joint not named here holds 0.0.
NOMINAL_POSE = {
    "left_hip_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.40,
    "right_knee_joint": 0.40,
    "left_ankle_pitch_joint": -0.20,
    "right_ankle_pitch_joint": -0.20,
}

FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
BASE_LINK = "pelvis"

# --- head -----------------------------------------------------------------
# H1-2 has no head link: the head is the top of the torso mesh. Points are in
# the torso link's frame.
HEAD_LINK = "torso_link"
CROWN = (0.045, 0.0, 0.76)     # top of the head
FOREHEAD = (0.110, 0.0, 0.70)  # where a header should land
# A ball contact on the torso counts as a header only above the neck.
NECK_Z = 0.50

# --- plant ----------------------------------------------------------------
DT = 0.002
ARMATURE = 0.02
DAMPING = 0.2
KP = 1000.0
KD = 20.0

SETTLE_SECONDS = 1.5
# Well past the landing, so a trajectory cannot score by toppling just after
# the clock stops.
HORIZON_S = 6.0

# --- the ball and the cross -----------------------------------------------
# A size-5 football.
BALL_RADIUS = 0.11
BALL_MASS = 0.43
BALL_FRICTION = 0.6
# Genesis contacts have no restitution coefficient; bounce comes from an
# underdamped contact constraint instead. A contact averages its two geoms'
# parameters, so the ball's own damping alone cannot make it lively: the head
# is softened too. Lower than this and the contact starts adding energy, which
# the search would exploit. The floor is left alone, since the feet share it.
BALL_DAMPRATIO = 0.02
HEAD_DAMPRATIO = 0.3

# The cross is described by how it arrives, and its launch is solved backward
# from that. The arrival point is the ball centre relative to the settled
# crown, in the world frame; the ball travels toward the robot (-x), descending.
BALL_OFFSET = (0.30, 0.0, 0.40)
ARRIVAL_SPEED = 6.0         # m/s
ARRIVAL_DESCENT_DEG = 20.0  # below horizontal
ARRIVAL_TIME = 1.5          # seconds into the rollout

# --- the goal -------------------------------------------------------------
# A regulation goal facing the robot, drawn but not collided with. The header
# is scored on where the ball crosses its plane.
GOAL_X = 7.0
GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44
GOAL_POST = 0.12           # post and crossbar thickness
TARGET = (0.0, 1.0)        # (y, z) in the goal plane
TARGET_SIZE = 0.6          # side of the drawn target square; display only
TARGET_LINE = 0.04         # its outline thickness

# --- the trajectory -------------------------------------------------------
# home -(T_wait)-> home -> anticipation -> stretch -> jump -> contact
#      -> overshoot -> home -> home
FRAMES = ("anticipation", "stretch", "jump", "contact", "overshoot")
SEGMENTS = ("anticipation", "stretch", "jump", "contact", "overshoot", "recover")

# A hand-written countermovement: the starting point of every cold search.
# Each pose is (hip_pitch, knee, ankle_pitch, shoulder_pitch) in radians.
SEED_POSES = {
    "anticipation": (-1.30, 1.50, -0.60, 0.50),
    "stretch": (-0.30, 0.20, 0.50, -1.00),
    "jump": (-0.20, 0.40, 0.20, -1.00),
    "contact": (-0.30, 0.60, 0.30, -0.50),
    "overshoot": (-1.50, 1.50, -0.60, -0.50),
}
SEED_DURATIONS = {  # seconds
    "anticipation": 0.25,
    "stretch": 0.15,
    "jump": 0.30,
    "contact": 0.30,
    "overshoot": 0.10,
    "recover": 0.50,
}
# Seconds holding home before the anticipation starts. The seed's is set so its
# `contact` knot falls on the ball's arrival.
SEED_WAIT = ARRIVAL_TIME - sum(SEED_DURATIONS[s] for s in SEGMENTS[:4])

# --- search box -----------------------------------------------------------
# A margin around the start point, not the joint limits: an unconstrained search
# deletes the jump, since standing still never falls.
FRAME_MARGIN = {frame: 0.30 for frame in FRAMES}  # radians
DURATION_MARGIN = {  # seconds
    "anticipation": 0.10,
    "stretch": 0.05,
    "jump": 0.05,
    "contact": 0.05,
    "overshoot": 0.05,
    "recover": 0.10,
}
# The wait is searched over its whole range, not a margin: lining the jump up
# with the ball is what it is for.
WAIT_RANGE = (0.0, 1.0)
# Keeps knot times strictly increasing when warm starts re-centre the box on a
# trajectory whose segments are already short.
MIN_DURATION = 0.02

PARAM_NAMES = (
    tuple(f"{frame}_{joint}" for frame in FRAMES for joint in CONTROLLED)
    + tuple(f"T_{segment}" for segment in SEGMENTS)
    + ("T_wait",)
)
SEED = (
    tuple(v for frame in FRAMES for v in SEED_POSES[frame])
    + tuple(SEED_DURATIONS[segment] for segment in SEGMENTS)
    + (SEED_WAIT,)
)

N_FRAMES = len(FRAMES)
N_JOINTS = len(CONTROLLED)
N_POSE_PARAMS = N_FRAMES * N_JOINTS
N_DURATIONS = len(SEGMENTS)
N_PARAMS = len(PARAM_NAMES)

# --- rendering ------------------------------------------------------------
# Frames are a whole number of steps apart, so pick a rate DT divides evenly at
# every VIDEO_SPEED in use.
RENDER_FPS = 50
# Playback speed of the videos: 0.5 is half speed.
VIDEO_SPEED = 0.5
RENDER_RES = (1280, 720)
# From behind the robot and to its side: the jump, the cross and the goal mouth
# in one frame. A pure side view sees the goal plane edge-on.
CAMERA_POS = (-2.5, -4.5, 2.4)
CAMERA_LOOKAT = (4.0, 0.0, 1.0)
CAMERA_FOV = 45
# A second view from behind the goal, looking back through the target at the
# robot: where the ball enters is easiest to read from here.
# Aimed between the target and the robot, and high enough to keep the ball's
# whole arc in frame.
GOAL_CAMERA_POS = (10.0, 2.2, 1.4)
GOAL_CAMERA_LOOKAT = (4.5, -0.8, 1.6)

# A key light plus a fill from the camera's side; the robot is dark metal and
# renders nearly black under the default single light. `dir` is the direction
# the light travels.
AMBIENT_LIGHT = (0.45, 0.45, 0.48)
LIGHTS = (
    {"type": "directional", "dir": (-1.0, -1.0, -1.0),
     "color": (1.0, 1.0, 1.0), "intensity": 5.0},
    {"type": "directional", "dir": (0.3, 1.0, -0.8),
     "color": (1.0, 0.98, 0.95), "intensity": 4.5},
)
BALL_COLOR = (0.95, 0.45, 0.10)
GOAL_COLOR = (0.95, 0.95, 0.95)
TARGET_COLOR = (0.15, 0.75, 0.30)
