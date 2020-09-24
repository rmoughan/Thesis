from agents.base import BaseAgent
from networks.convNet import ConvNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class Actor_Critic(BaseAgent):

    def __init__(self, config, logger, device):
        super().__init__(config, logger, device)
        self.actor = ConvNet(self.actor_architecture_config)
        self.actor.to(self.device)
        self.critic = ConvNet(self.critic_architecture_config)
        self.critic.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.lr._v) #may want to change eps
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.lr._v) #may want to change eps
        #may want to change gamma, I think their's is 1.0
        self.gamma = 1.0

    def collect_trajectories(self):
        total_timesteps_this_batch = 0
        trajectories = []
        while total_timesteps_this_batch < self.batch_size:
            trajectory = self._collect_trajectory()
            trajectories.append(trajectory)
            self.logger.add_reward(np.sum(trajectory["reward"]))
            total_timesteps_this_batch += len(trajectory["reward"])


        if self.t == 0:
            import pickle
            infile = open("./debugging/paths", "rb")
            paths = pickle.load(infile)
            for p in paths:
                p["done"] = p["terminal"]
            total_timesteps_this_batch = sum([len(p["reward"]) for p in paths])
            trajectories = paths
            infile.close()

        self.total_timesteps += total_timesteps_this_batch
        self.replay_buffer.add_rollouts(trajectories)


    def _collect_trajectory(self):
        obs, acs, rewards, next_obs, dones = [], [], [], [], []
        ob = self.env.reset()
        steps = 0
        while True:
            obs.append(ob)
            ob = torch.from_numpy(ob).float()
            logits = self.actor(ob)
            #start debugging from here. So it looks like the logits
            #are the same, I'm still not entirely clear what's going on with the
            #lack of softmax in the 285 code though
            #Added softmax which seems fine but we will see
            ac = torch.multinomial(F.softmax(logits), num_samples = 1).item()
            acs.append(ac)

            ob, reward, done, _ = self.env.step(ac)
            steps += 1

            next_obs.append(ob)
            rewards.append(reward)
            trajectory_done = 1 if done or len(rewards) == self.max_path_length else 0
            dones.append(trajectory_done)
            if trajectory_done:
                break

        return {"observation" : np.array(obs, dtype=self.obs_dtype),
                "action" : np.array(acs, dtype=np.float32),
                "reward" : np.array(rewards, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=self.obs_dtype),
                "done": np.array(dones, dtype=np.float32)}


    def update(self):

        #need an outer for loop here or just to rewrite the training loop for on policy
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample_recent_data(self.batch_size) #may want to be different batch size
        obs_batch = torch.from_numpy(obs_batch).float()
        act_batch = torch.from_numpy(act_batch).float()
        rew_batch = torch.from_numpy(rew_batch).float()
        next_obs_batch = torch.from_numpy(next_obs_batch).float()
        done_mask = torch.from_numpy(done_mask).float()

        #update critic
        torch.autograd.set_detect_anomaly(True)
        # for _ in range(self.num_critic_updates_per_agent_update):
        #     for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
        #         if i % self.num_grad_steps_per_target_update == 0:
        next_v_values = self.critic(next_obs_batch).squeeze()
        target_v_values = rew_batch + self.gamma * next_v_values * (1 - done_mask)
        predicted_v_values = self.critic(obs_batch).squeeze()
        critic_loss = nn.MSELoss()(predicted_v_values, target_v_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #gradients are vastly different, so I need to figure out how
        #to fix that somehow. I honestly don't know how to fix this at this
        #exact moment in time so I may start on the hierarchical algorithms and
        #then come back because 61a
        for p in self.critic.parameters():
            print(p.data)
            print(p.grad.data)
            print("-------------------------")
        pdb.set_trace()
        # possibly clip gradients
        self.critic_optimizer.step()

        #update is still different, so I should check if the gradients are
        #different when I have the energy
        #changing to torch 1.4 "fixed it"...

        #estimate advantages
        v_values = self.critic(obs_batch).squeeze()
        next_v_values = self.critic(next_obs_batch).squeeze()
        q_values = rew_batch + self.gamma * next_v_values * (1 - done_mask)
        adv_batch = q_values - v_values
        adv_batch = (adv_batch - torch.mean(adv_batch)) / (torch.std(adv_batch) + 1e-8)


        #update actor (currently only works for discrete action space)
        for _ in range(self.num_actor_updates_per_agent_update):
            logits = self.actor(obs_batch)
            log_prob = torch.distributions.Categorical(logits).log_prob(act_batch)
            actor_loss = torch.sum(-1 * log_prob * adv_batch)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            #possibly clip gradients
            self.actor_optimizer.step()
