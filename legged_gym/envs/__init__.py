from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

# from legged_gym.envs.lite3.lite3_config import Lite3RoughCfg,LeggedRobotCfgPPO
from .base.legged_robot_reset import LeggedRobotReset
from .base.legged_robot import LeggedRobot
from .lite3.lite3_parkour_config import Lite3ParkourCfg,Lite3ParkourCfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register("lite3",LeggedRobot,Lite3ParkourCfg(),Lite3ParkourCfgPPO())
task_registry.register("lite3_reset",LeggedRobotReset,Lite3ParkourCfg(),Lite3ParkourCfgPPO())

