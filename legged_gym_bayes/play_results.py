import csv
from legged_gym.envs import *
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import update_cfg_from_args,class_to_dict,parse_sim_params,get_load_path

import torch
import numpy as np
from collections import deque
from legged_gym_learn.runners.on_policy_runner import OnPolicyRunner
import statistics

class Play():
    def __init__(self,args):
        self.args = args
        self.log_path = "../logs/{}/".format(self.args.proj_name) + self.args.exptid
        self.env_cfg,self.train_cfg = task_registry.get_cfgs(name=self.args.task)
        self.env_cfg,_ = update_cfg_from_args(self.env_cfg,None,self.args)
        self.sim_params = {"sim":class_to_dict(self.env_cfg.sim)}
        self.sim_params = parse_sim_params(self.args,self.sim_params)
        _,self.train_cfg = update_cfg_from_args(None,self.train_cfg,self.args)
        self.train_cfg_dict = class_to_dict(self.train_cfg)
        self._reset_env_cfg()
        self.optim_resume_checkpoint = 400
        self.design_idx = 0

    def read_file(self,path):
        with open(path,'r') as fd:
            cereader = csv.reader(fd)
            data = []
            skipped_first_row = False
            for row in cereader:
                if not skipped_first_row:
                    skipped_first_row = True
                    continue
                row_tmp = row
                for idx in range(len(row_tmp)):
                    row_tmp[idx] = float(row_tmp[idx])
                data.append(row_tmp)
            designs = data[0]
            return designs


    def _reset_env_cfg(self):
        if self.args.nodelay:
            self.env_cfg.domain_rand.action_delay_view = 0
        self.env_cfg.env.num_envs = 1
        self.env_cfg.env.episode_length_s = 20
        self.env_cfg.commands.resampling_time = 20  # 6.
        self.env_cfg.terrain.num_rows = 2           # 10
        self.env_cfg.terrain.num_cols = 2          # 40
        self.env_cfg.terrain.height = [0.02,0.02]   # [0.02,0.06]
        self.env_cfg.terrain.terrain_dict = {"smooth slope": 0.0,
                                              "rough slope up": 0.0,
                                              "rough slope down": 0.0,
                                              "rough stairs up": 0.0,
                                              "rough stairs down": 0.0,
                                              "discrete": 0.0,
                                              "stepping stones": 0.0,
                                              "gaps": 0.0,
                                              "smooth flat": 0.0,
                                              "pit": 0.0,
                                              "wall": 0.0,
                                              "platform": 0.0,
                                              "large stairs up": 0.0,
                                              "large stairs down": 0.0,
                                              "parkour": 0.0,
                                              "parkour_hurdle": 0.0,    # jump high
                                              "parkour_flat": 0.0,
                                              "parkour_step": 0.0,
                                              "parkour_gap": 1.0,      # jump long
                                              "demo": 0.0}
        self.env_cfg.terrain.terrain_proportions = list(self.env_cfg.terrain.terrain_dict.values())
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.terrain.max_difficulty = True
        self.env_cfg.terrain.manual_set = True

        self.env_cfg.depth.angle = [0, 1]
        self.env_cfg.noise.add_noise = True
        self.env_cfg.domain_rand.randomize_friction = True
        self.env_cfg.domain_rand.push_robots = False
        self.env_cfg.domain_rand.push_interval_s = 6
        self.env_cfg.domain_rand.randomize_base_mass = False
        self.env_cfg.domain_rand.randomize_base_com = False
        self.env_cfg.domain_rand.action_delay = False

        ''' parkour_gap '''
        gap_difficulty = 1.0
        self.env_cfg.terrain.gap_gap_size = gap_difficulty
        self.env_cfg.terrain.gap_depth = [0.9, 1.0]
        self.env_cfg.terrain.gap_x_range = [0.8, 1.5]
        self.env_cfg.terrain.gap_y_range = [-0.1, 0.1]
        self.env_cfg.terrain.gap_half_valid_width = [0.6, 1.2]
        ''' parkour_hurdle '''
        max_difficulty = np.random.uniform(0.9, 1.0)
        self.env_cfg.terrain.hurdle_stone_len = 0.2 + 0.3 * max_difficulty  # [0.37,0.4]
        self.env_cfg.terrain.hurdle_height_range = [0.565, 0.575]
        self.env_cfg.terrain.hurdle_x_range = [2.5, 3.0]  # [1.2, 2.2]
        self.env_cfg.terrain.hurdle_y_range = [-0.1, 0.1]  # [-0.4,0.4]
        self.env_cfg.terrain.hurdle_half_valid_width = [0.8, 1.6]


    def evaluate(self):
        '''make env'''
        self.args.headless = False
        self.path = args.exptid + '/best_design.csv'
        self.current_designs= self.read_file(path=self.path)
        self.env_cfg.designs.specified_design = np.array(self.current_designs)
        self.env = LeggedRobotReset(cfg=self.env_cfg,sim_params=self.sim_params,physics_engine=self.args.physics_engine,sim_device=self.args.sim_device,headless=self.args.headless)
        '''make runner'''
        self.current_log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(self.args.proj_name) + self.args.exptid + '/' + "designs_" + str(self.design_idx)
        self.runner = OnPolicyRunner(env=self.env,train_cfg=self.train_cfg_dict,log_dir=self.current_log_pth,device=self.args.rl_device,
                                     record=False,design_iterations=None,designs=None,max_iterations=None)
        self.fine_tune_path = get_load_path(self.current_log_pth,checkpoint=self.optim_resume_checkpoint)
        self.runner.load(self.fine_tune_path)
        self.policy = self.runner.get_inference_policy(device=self.env.device)
        '''run the policy'''
        obs = self.env.get_observations()
        self.total_eval_steps = 1 * int(self.env.max_episode_length)
        rewbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs,dtype=torch.float,device=self.env.device)
        for i in range(self.total_eval_steps):
            actions = self.policy(obs.detach(),hist_encoding=False,scandots_latent=None)
            obs,_,rews,dones,infos = self.env.step(actions.detach())
            rews = rews.to(self.env.device).clone()
            dones = dones.to(self.env.device)
            cur_reward_sum += rews
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:,0].cpu().numpy().tolist())
        mean_reward = statistics.mean(rewbuffer)
        self.env.gym.destroy_viewer(self.env.viewer)
        self.env.gym.destroy_sim(self.env.sim)
        return mean_reward


if __name__ == "__main__":
    args = get_args()
    args.exptid = "exp_name"
    play = Play(args=args)
    args.seed = 10
    play.evaluate()



