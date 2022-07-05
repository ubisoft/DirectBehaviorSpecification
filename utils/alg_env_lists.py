import arenaEnv.arenaEnvPython
from pathlib import Path
import sys

# Algorithms

ON_POLICY_ALGS = set()
OFF_POLICY_ALGS = {"sac", "td3"}

ALL_ALGS = ON_POLICY_ALGS | OFF_POLICY_ALGS

# Environments

CLASSIC_ENVS = set()
MUJOCO_ENVS = set()
UNITY_ENVS = {"ArenaEnv-v0"}
CONSTRAINED_ENVS = {"ArenaEnv-v0"}

ALL_ENVS = CLASSIC_ENVS | MUJOCO_ENVS | UNITY_ENVS

# Unity envs info

if sys.platform == "linux":
    build_folder = "build_linux"
elif sys.platform == "win32":
    build_folder = "build_windows"
else:
    raise ValueError("Unsupported platform.")

TASK_NAME_TO_EXEC_PATH = {
    "ArenaEnv-v0": str(Path(arenaEnv.arenaEnvPython.__file__).parent / f"ArenaEnv/{build_folder}/arenaEnvUnityProject")
}

TASK_NAME_TO_MAX_EPISODE_STEPS = {
    "ArenaEnv-v0": 200
}