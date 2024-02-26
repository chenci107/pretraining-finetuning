import copy
import csv
import os

import numpy as np
from legged_gym.envs import *
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import update_cfg_from_args,set_seed,class_to_dict,parse_sim_params,get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR,LEGGED_GYM_ENVS_DIR
from legged_gym_learn.runners.on_policy_runner import OnPolicyRunner
from hyperopt import fmin,tpe,hp,Trials
import torch
from collections import deque
import statistics

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
CURRENT_PATH = os.path.abspath(os.path.join(os.getcwd()))

FINETUNE_ITERATIONS = 401
BAYES_ITERATIONS = 30

def read_file(path):
    with open(path,'r') as fd:
        cwreader = csv.reader(fd)
        design = []
        skipped_first_row = False
        for row in cwreader:
            if not skipped_first_row:
                skipped_first_row = True
                continue
            row_tmp = row
            for idx in range(len(row_tmp)):
                row_tmp[idx] = float(row_tmp[idx])
            design.append(row_tmp)

        return design[0]

class BayesOpt():
    def __init__(self,args,design_iter=0):
        self.args = args
        self.train_env_cfg,self.train_cfg = task_registry.get_cfgs(name=self.args.task)
        '''train_env_cfg ==> self.sim_params'''
        self.train_env_cfg,_ = update_cfg_from_args(self.train_env_cfg,None,self.args)
        self._reset_train_env_cfg()
        set_seed(self.train_env_cfg.seed)
        self.train_cfg.runner.save_interval = int((FINETUNE_ITERATIONS-1) / 2)
        self.train_cfg.runner.max_iterations = FINETUNE_ITERATIONS
        self.optim_resume_checkpoint = int(FINETUNE_ITERATIONS-1)
        self.base_resume_checkpoint = 5500
        self.sim_params = {"sim":class_to_dict(self.train_env_cfg.sim)}
        self.sim_params = parse_sim_params(self.args,self.sim_params)
        '''test_env_cfg'''
        self.test_env_cfg = copy.deepcopy(self.train_env_cfg)
        self._reset_test_env_cfg()
        '''train_cfg ==> self.train_cfg_dict'''
        _,self.train_cfg = update_cfg_from_args(None,self.train_cfg,self.args)
        self.train_cfg_dict = class_to_dict(self.train_cfg)
        self.base_model_path =  LEGGED_GYM_ROOT_DIR + f"/logs/{self.args.proj_name}/" + self.args.resumeid
        self.base_model_path = get_load_path(self.base_model_path,checkpoint=self.base_resume_checkpoint)
        '''some hyper-parameters'''
        self.design_iterations = design_iter
        '''make record dirs'''
        self.record_dir = CURRENT_PATH + "/" + self.args.exptid
        try:
            os.makedirs(self.record_dir)
        except:
            pass

    def fine_tune(self,design):
        self.args.headless = True
        self.current_log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(self.args.proj_name) + self.args.exptid + "/" + "designs_" + str(self.design_iterations)
        try:
            os.makedirs(self.current_log_pth)
        except:
            pass
        '''make env'''
        self.train_env_cfg.designs.specified_design = np.array([design[0],design[1],design[2],design[3]])
        self.train_env = LeggedRobotReset(cfg=self.train_env_cfg,sim_params=self.sim_params,physics_engine=self.args.physics_engine,sim_device=self.args.sim_device,headless=self.args.headless)
        '''make runner'''
        self.train_runner = OnPolicyRunner(env=self.train_env,train_cfg=self.train_cfg_dict,log_dir=self.current_log_pth,device=self.args.rl_device,
                                           record=True,design_iterations=self.design_iterations,designs=design,max_iterations=FINETUNE_ITERATIONS,exptid=self.args.exptid)
        self.train_runner.load(self.base_model_path)
        '''start learning'''
        self.train_runner.learn_RL(num_learning_iterations=self.train_cfg.runner.max_iterations,init_at_random_ep_len=True)
        '''destroy the env'''
        self.design_iterations += 1
        self.train_env.gym.destroy_viewer(self.train_env.viewer)
        self.train_env.gym.destroy_sim(self.train_env.sim)

    def evaluate(self,design):
        '''make env'''
        self.args.headless = True
        self.test_env_cfg.designs.specified_design = np.array([design[0],design[1],design[2],design[3]])
        self.test_env = LeggedRobotReset(cfg=self.test_env_cfg,sim_params=self.sim_params,physics_engine=self.args.physics_engine,sim_device=self.args.sim_device,headless=self.args.headless)
        '''make runner'''
        self.test_runner = OnPolicyRunner(env=self.test_env,train_cfg=self.train_cfg_dict,log_dir=self.current_log_pth,device=self.args.rl_device,record=False)
        self.fine_tune_path = get_load_path(self.current_log_pth,checkpoint=self.optim_resume_checkpoint)
        self.test_runner.load(self.fine_tune_path)
        self.policy = self.test_runner.get_inference_policy(device=self.test_env.device)
        self.estimator = self.test_runner.get_estimator_inference_policy(device=self.test_env.device)
        if self.test_env.cfg.depth.use_camera:
            self.depth_encoder = self.test_runner.get_depth_encoder_inference_policy(device=self.test_env.device)
        '''run the policy'''
        obs = self.test_env.get_observations()
        rewbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.test_env.num_envs,dtype=torch.float,device=self.test_env.device)
        for i in range(1 * int(self.test_env.max_episode_length)):
            actions = self.policy(obs.detach(),hist_encoding=False,scandots_latent=None)
            obs,_,rews,dones,infos = self.test_env.step(actions.detach())
            rews = rews.to(self.test_env.device).clone()
            dones = dones.to(self.test_env.device)
            cur_reward_sum += rews
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:,0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0

        mean_reward = statistics.mean(rewbuffer)
        fitness =  - mean_reward  # important
        '''destroy the env'''
        self.test_env.gym.destroy_viewer(self.test_env.viewer)
        self.test_env.gym.destroy_sim(self.test_env.sim)
        if self.design_iterations < BAYES_ITERATIONS:
            with open(self.record_dir + "/fitness.csv",'a',newline='') as file_obj:
                writer = csv.writer(file_obj)
                data = [self.design_iterations,fitness]
                writer.writerow(data)
        else:
            with open(self.record_dir + "/best_design.csv",'a',newline='') as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow([fitness])
        return fitness


    def cal_objective(self,params):
        front_upper_scale = params["front_upper_ratio"]
        front_lower_scale = params["front_lower_ratio"]
        hind_upper_scale = params["hind_upper_ratio"]
        hind_lower_scale = params["hind_lower_ratio"]

        design = np.array([front_upper_scale,front_lower_scale,hind_upper_scale,hind_lower_scale])
        '''fine-tune the policy'''
        self.fine_tune(design)
        '''calculate the fitness'''
        fitness = self.evaluate(design)
        return fitness

    def optimize_leg(self):
        fspace = {
            "front_upper_ratio":hp.uniform("front_upper_ratio",0.6,1.4),
            "front_lower_ratio":hp.uniform("front_lower_ratio",0.6,1.4),
            "hind_upper_ratio":hp.uniform("hind_upper_ratio",0.6,1.4),
            "hind_lower_ratio":hp.uniform("hind_lower_ratio",0.6,1.4),
        }

        trials = Trials()
        best = fmin(
            fn = self.cal_objective,
            space = fspace,
            algo = tpe.suggest,
            max_evals = BAYES_ITERATIONS,
            trials = trials,)

        header = ["front_upper_ratio","front_lower_ratio","hind_upper_ratio","hind_lower_ratio"]
        datas = [best]
        with open(self.record_dir + "/best_design.csv",'w',newline='',encoding='utf-8') as file_obj:
            writer = csv.DictWriter(file_obj,fieldnames=header)
            writer.writeheader()
            writer.writerows(datas)

        print('The best design is:',best)


    def _reset_test_env_cfg(self):
        if self.args.nodelay:
            self.test_env_cfg.domain_rand.action_delay_view = 0
        self.test_env_cfg.env.num_envs = 64
        self.test_env_cfg.env.episode_length_s = 20
        self.test_env_cfg.commands.resampling_time = 20  # 6.
        self.test_env_cfg.terrain.num_rows = 5           # 10
        self.test_env_cfg.terrain.num_cols = 5           # 40
        self.test_env_cfg.terrain.height = [0.02,0.02]   # [0.02,0.06]
        self.test_env_cfg.terrain.terrain_dict = {"smooth slope": 0.0,
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
                                                  "parkour_hurdle": 1.0,    # jump high
                                                  "parkour_flat": 0.0,
                                                  "parkour_step": 0.0,
                                                  "parkour_gap": 0.0,      # jump long
                                                  "demo": 0.0}
        self.test_env_cfg.terrain.terrain_proportions = list(self.test_env_cfg.terrain.terrain_dict.values())
        self.test_env_cfg.terrain.curriculum = False
        self.test_env_cfg.terrain.max_difficulty = True
        self.test_env_cfg.terrain.manual_set = True

        self.test_env_cfg.depth.angle = [0, 1]
        self.test_env_cfg.noise.add_noise = True
        self.test_env_cfg.domain_rand.randomize_friction = True
        self.test_env_cfg.domain_rand.push_robots = False
        self.test_env_cfg.domain_rand.push_interval_s = 6
        self.test_env_cfg.domain_rand.randomize_base_mass = False
        self.test_env_cfg.domain_rand.randomize_base_com = False
        self.test_env_cfg.domain_rand.action_delay = False

        '''parkour_hurdle related'''
        self.eval_hurdle_difficulty = copy.deepcopy(self.train_hurdle_difficulty)
        self.test_env_cfg.terrain.hurdle_stone_len = copy.deepcopy(self.train_env_cfg.terrain.hurdle_stone_len)
        self.test_env_cfg.terrain.hurdle_height_range = copy.deepcopy(self.train_env_cfg.terrain.hurdle_height_range)
        self.test_env_cfg.terrain.hurdle_x_range = copy.deepcopy(self.train_env_cfg.terrain.hurdle_x_range)
        self.test_env_cfg.terrain.hurdle_y_range = copy.deepcopy(self.train_env_cfg.terrain.hurdle_y_range)
        self.test_env_cfg.terrain.hurdle_half_valid_width = copy.deepcopy(self.train_env_cfg.terrain.hurdle_half_valid_width)
        '''parkour_gap related'''
        self.eval_gap_difficulty = copy.deepcopy(self.train_gap_difficulty)
        self.test_env_cfg.terrain.gap_gap_size = copy.deepcopy(self.train_env_cfg.terrain.gap_gap_size)
        self.test_env_cfg.terrain.gap_depth = copy.deepcopy(self.train_env_cfg.terrain.gap_depth)
        self.test_env_cfg.terrain.gap_x_range = copy.deepcopy(self.train_env_cfg.terrain.gap_x_range)
        self.test_env_cfg.terrain.gap_y_range = copy.deepcopy(self.train_env_cfg.terrain.gap_y_range)
        self.test_env_cfg.terrain.half_valid_width = copy.deepcopy(self.train_env_cfg.terrain.half_valid_width)


    def _reset_train_env_cfg(self):
        self.train_env_cfg.env.num_envs = 2048
        self.train_env_cfg.terrain.terrain_dict = {"smooth slope": 0.0,
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
                                              "parkour_hurdle": 1.0,  # jump high
                                              "parkour_flat": 0.0,
                                              "parkour_step": 0.0,
                                              "parkour_gap": 0.0,  # jump long
                                              "demo": 0.0}

        self.train_env_cfg.terrain.terrain_proportions = list(self.train_env_cfg.terrain.terrain_dict.values())
        self.train_env_cfg.terrain.curriculum = False
        self.train_env_cfg.terrain.max_difficulty = True
        self.train_env_cfg.terrain.manual_set = True

        '''parkour_hurdle related'''
        self.train_hurdle_difficulty = np.random.uniform(0.9, 1.0)
        self.train_env_cfg.terrain.hurdle_stone_len = 0.2 + 0.3 * self.train_hurdle_difficulty  # [0.37,0.40]
        self.train_env_cfg.terrain.hurdle_height_range = [0.57,0.58]
        self.train_env_cfg.terrain.hurdle_x_range = [2.5,3.0]
        self.train_env_cfg.terrain.hurdle_y_range = [-0.1,0.1]
        self.train_env_cfg.terrain.hurdle_half_valid_width = [0.8,1.6]
        '''parkour_gap related'''
        self.train_gap_difficulty = np.random.uniform(0.7, 0.9)
        self.train_env_cfg.terrain.gap_gap_size = self.train_gap_difficulty
        self.train_env_cfg.terrain.gap_depth = [0.9,1.0]
        self.train_env_cfg.terrain.gap_x_range = [0.8, 1.5]
        self.train_env_cfg.terrain.gap_y_range = [-0.1, 0.1]
        self.train_env_cfg.terrain.half_valid_width = [0.6, 1.2]

    def read_design_and_perform(self):
        '''read the design'''
        file_path = self.record_dir + "/best_design.csv"
        design = read_file(file_path)
        design = np.array([design[0],design[1],design[2],design[3]])
        '''fine-tune the policy'''
        self.fine_tune(design)
        '''calculate the fitness'''
        fitness = self.evaluate(design)
        print('='*50)
        print('The fitness is:',fitness)
        print('='*50)


if __name__ == "__main__":
    args = get_args()
    args.seed = 10
    args.proj_name = 'parkour_new'
    args.exptid = "exp_name"
    args.resumeid = ""
    '''=== step 1 ==='''
    Bayes_opt = BayesOpt(args=args,design_iter=0)
    Bayes_opt.optimize_leg()
    '''=== step 2 ==='''
    # Bayes_opt = BayesOpt(args=args,design_iter=BAYES_ITERATIONS)
    # Bayes_opt.read_design_and_perform()










