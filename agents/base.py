import torch
import torch.nn as nn
from infrastructure.off_policy_buffer import OffPolicyReplayBuffer
from infrastructure.on_policy_buffer import OnPolicyReplayBuffer
from infrastructure.config import Config
from infrastructure.utils import sow_field
from envs.atari_wrappers import wrap_deepmind
from envs.montezuma_wrapper import MontezumaWrapper
import gym
import numpy as np
from infrastructure.scheduler import *
from cv2 import VideoWriter, VideoWriter_fourcc


import pdb

class BaseAgent:

    def __init__(self, config, logger, device):

        self.logger = logger
        self.device = device
        self.config = config
        self.plot_experiment = config.params.get("plot_value", False) is not False
        self.experiment_filepath = "./experiments/" + self.config.params["name"] + "/" + "seed" + "_" + str(self.config.params["seed"])
        self.env_is_gym = config.params["env_is_gym"]
        self.wrap_env = config.params["wrap_env"]
        if self.env_is_gym:
            self.env = gym.make(config.params["env_name"])
            self.cartpole = True if config.params["env_name"] == "CartPole-v0" else False
            if self.wrap_env:
                self.env = wrap_deepmind(self.env)
            if 'MontezumaRevenge' in config.params["env_name"]:
                self.env = MontezumaWrapper(self.env)
        else:
            raise ValueError("Non-gym environments not supported yet")
        self.seed = config.params.get("seed", None)
        if self.seed != None:
            sow_field(self.seed, self.env)
        self.lr = eval(config.params["lr"])
        self.hierarchical = config.params.get("hierarchical", 0) == 1
        self.on_policy = config.params["model"] != "DQN"
        if self.on_policy:
            self.actor_architecture_config = Config(config.params["actor_architecture_config"])
            self.critic_architecture_config = Config(config.params["critic_architecture_config"])
            self.num_iterations = config.params.get("num_iterations", 200)
            self.batch_size = config.params.get("batch_size", 1000)
            self.total_timesteps = 0
            self.max_path_length = config.params.get("episode_length", 200)
            self.num_critic_updates_per_agent_update = config.params.get("num_critic_updates_per_agent_update", 1)
            self.num_actor_updates_per_agent_update = config.params.get("num_actor_updates_per_agent_update", 1)
            self.num_grad_steps_per_target_update = config.params.get("num_grad_steps_per_target_update", 10)
            self.num_target_updates = config.params.get("num_target_updates", 10)
            self.replay_size = config.params.get("replay_size", 1000000)
            self.replay_buffer = OnPolicyReplayBuffer(self.replay_size)
            self.obs_dtype = np.float32 if self.cartpole else np.uint8
            self.report_freq = config.params.get("report_freq", self.num_iterations / 100)
            self.train = self.train_on_policy

        else:
            self.network_architecture_config = Config(config.params["network_architecture_config"])
            self.num_timesteps = config.params["num_timesteps"]
            self.max_episode_length = config.params.get("max_episode_length", 200) #this is redundant with max_path_length in on_policy
            self.batch_size = config.params.get("batch_size", 32)
            self.epsilon = eval(config.params["epsilon"])
            self.learning_starts = config.params.get("learning_starts", 50000)
            self.learning_freq = config.params.get("learning_freq", 4)
            self.target_update_freq = config.params.get("target_update_freq", 10000)
            self.replay_size = config.params.get("replay_size", 1000000)
            self.frame_history_len = config.params.get("frame_history_len", 4)
            self.replay_buffer = OffPolicyReplayBuffer(self.replay_size, self.frame_history_len, self.cartpole, self.hierarchical)
            self.num_param_updates = 0
            self.report_freq = config.params.get("report_freq", self.num_timesteps / 100)
            self.train = self.train_off_policy
            self.train = self.montezuma_train_off_policy2 #change back later
        self.record_freq = config.params.get("record_freq", 0) #for simplicity, I'm going to put this frequency in terms of episodes
        self.recording = False
        self.momentum = config.params.get("momentum", 0)
        self.alpha = config.params.get("alpha", 0.99)
        self.gamma = config.params.get("gamma", 0.99)
        self.t = 0
        self.save_weights = config.params.get("save_weights", False)
        self.debugging = config.params.get("debugging_fuck_this", False)

    def train_off_policy(self):
        print("training")
        self.last_obs = self.env.reset()
        while self.t < self.num_timesteps:
            self.done = False
            self.episode_reward = 0
            while not self.done:
                self.step()
                self.update()
                if self.t % self.report_freq == 0:
                    self.logger.report(self.t, self.epsilon.value(self.t))
            self.logger.add_reward(self.episode_reward)
        if self.plot_experiment:
            self.logger.graph()
        self.logger.save_experiment()

    def train_on_policy(self):
        print("training")
        while self.t <= self.num_iterations:
            self.collect_trajectories()
            #I could put a for loop right here, I have iffy opinions about it
            self.update()
            if self.t % self.report_freq == 0:
                self.logger.report(self.t, self.epsilon.value(self.t))
            self.t += 1
        if self.plot_experiment:
            self.logger.graph()
        self.logger.save_experiment()

    def montezuma_train_off_policy(self): #for now I'm not using the deepmind wrapper
        """A training policy that is meant to test whether or not the controller works. It seems like it does,
        so this is now DEPRECATED.
        """
        print("training")

        self.last_obs = self.env.reset()
        self.env.current_goal = (111, 121)
        print("goal: ", self.env.current_goal)
        while self.t < self.num_timesteps:
            self.done, self.at_goal = False, False
            self.episode_reward = 0
            self.intrinsic_reward = 0
            self.episode_length = 0
            self.goals_reached = []
            while not self.done and not self.at_goal and self.episode_length <= self.max_episode_length:
                self.step()
                self.update()
                self.episode_length += 1
                if self.t % self.report_freq == 0:
                    self.logger.report(self.t, self.epsilon.value(self.t))
            if self.done or self.at_goal or self.episode_length >= self.max_episode_length:
                self.last_obs = self.env.reset()
                self.logger.add_reward(self.intrinsic_reward, self.goals_reached)
        if self.plot_experiment:
            self.logger.graph()
        self.logger.save_experiment()

    def montezuma_train_off_policy2(self):

        print("training")
        # self.goals = [(111, 121), (134.45714285714286, 123.57142857142857), (135, 155)] #(131.25833333, 168.26666667)
        #self.goals = [(75.47058824, 123.23529412), (88.52941176, 123.23529412), (65, 126.53846154)]
        self.goals = [(78, 127)]
        self.last_obs = self.env.reset()
        self.current_goal_index = 0
        self.env.current_goal = self.goals[self.current_goal_index]
        self.done, self.at_goal = False, False
        self.episode_reward, self.intrinsic_reward = 0, 0
        self.episode_length, self.goals_reached = 0, []
        self.num_episodes = 0
        print("goal: ", self.env.current_goal)
        while self.t < self.num_timesteps:
            while not self.done and not self.at_goal and self.episode_length <= self.max_episode_length:
                self.step()
                self.update()
                self.episode_length += 1
                if self.t % self.report_freq == 0:
                    failure_rates = [1 - (len([i for i in self.logger.goals_reached[-100:] if g in i]) / 100) for g in self.goals]
                    # self.logger.report(self.t, [round(self.epsilon.value(self.goals[i], failure_rates[i]), 3) for i in range(len(self.goals))])
                    self.logger.report(self.t, self.epsilon.value(self.t))
            if not self.done and self.episode_length <= self.max_episode_length and self.current_goal_index < len(self.goals) - 1:
                self.current_goal_index += 1
                self.env.current_goal = self.goals[self.current_goal_index]
                self.at_goal = False
            else:
                if self.recording:
                    self.recording = False
                    self.video.release()
                self.logger.add_reward(len(self.goals_reached), self.episode_length, self.goals_reached)
                self.num_episodes += 1
                if self.num_episodes % self.record_freq == 0:
                    video_filepath = self.experiment_filepath + '/episode_' + str(self.num_episodes) + "_recording" + ".mp4"
                    self.video = VideoWriter(video_filepath, 0x7634706d, 15.0, (160, 210), isColor = True)
                    self.recording = True
                self.last_obs = self.env.reset()
                self.done, self.at_goal = False, False
                self.episode_reward, self.intrinsic_reward = 0, 0
                self.episode_length, self.goals_reached = 0, []
                self.current_goal_index = 0
                self.env.current_goal = self.goals[self.current_goal_index]
        if self.plot_experiment:
            self.logger.graph()
        self.logger.save_experiment()
        # if self.save_weights:
        #     torch.save(self.current_network.state_dict(), self.experiment_filepath + "/current_network_weights")
        #     torch.save(self.target_network.state_dict(), self.experiment_filepath + "/target_network_weights")
