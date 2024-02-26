from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Lite3ParkourCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos =  [0.0, 0.0, 0.38]
        default_joint_angles = {
            'FL_HipX_joint': 0.,
            'FR_HipX_joint': 0.,
            'HL_HipX_joint': 0.,
            'HR_HipX_joint': 0.,

            'FL_HipY_joint': 0.72,
            'FR_HipY_joint': 0.72,
            'HL_HipY_joint': 0.72,
            'HR_HipY_joint': 0.72,

            'FL_Knee_joint': -1.45,
            'FR_Knee_joint': -1.45,
            'HL_Knee_joint': -1.45,
            'HR_Knee_joint': -1.45,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.0}  # [N*m/rad]
        damping = {'joint': 0.7}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/legged_gym/resources/lite3/urdf/Lite3+foot_changed.urdf'
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "shoulder", "SHANK"]
        terminate_after_contacts_on = ["TORSO", "shoulder"]
        self_collisions = 1

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.31

class Lite3ParkourCfgPPO(LeggedRobotCfgPPO):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_lite3'


