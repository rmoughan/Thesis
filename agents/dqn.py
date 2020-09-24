from agents.base import BaseAgent
from networks.convNet import ConvNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class DQN(BaseAgent):
    def __init__(self, config, logger, device):
        super().__init__(config, logger, device)
        self.current_network = ConvNet(self.network_architecture_config)
        self.current_network.to(self.device)
        self.target_network = ConvNet(self.network_architecture_config)
        self.target_network.load_state_dict(self.current_network.state_dict())
        self.target_network.to(self.device)
        self.optimizer = torch.optim.Adam(self.current_network.parameters(), lr = self.lr._v, eps = 5e-4)
        #eps is weird and need to potentially mess around with built in lr schedule class in pytorch

    def step(self):

        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        epsilon = self.epsilon.value(self.t)

        if np.random.random() <= epsilon or self.t <= self.learning_starts:
            action = self.env.action_space.sample()

        else:
            enc_last_obs = self.replay_buffer.encode_recent_observation()
            if self.cartpole:
                enc_last_obs = torch.from_numpy(enc_last_obs).unsqueeze(dim = 0).float().to(self.device)
            else:
                enc_last_obs = torch.from_numpy(enc_last_obs).unsqueeze(dim = 0).permute(0,3,1,2).float().to(self.device) / 255.0
            with torch.no_grad():
                q_values = self.current_network(enc_last_obs)
                action = q_values.argmax()
                if self.cartpole:
                    action = action.item()

        obs, reward, done, info = self.env.step(action)
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        self.episode_reward += reward
        if done:
            obs = self.env.reset()
        self.last_obs = obs
        self.done = done

    def update(self):

        if (self.t > self.learning_starts and \
            self.t % self.learning_freq == 0 and \
            self.replay_buffer.can_sample(self.batch_size)):

            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)
            obs_batch = torch.from_numpy(obs_batch).float().to(self.device)
            act_batch = torch.from_numpy(np.int64(act_batch)).unsqueeze(1).to(self.device)
            rew_batch = torch.from_numpy(rew_batch).to(self.device)
            next_obs_batch = torch.from_numpy(next_obs_batch).float().to(self.device)
            done_mask = torch.from_numpy(done_mask).to(self.device)

            if not self.cartpole:
                obs_batch = obs_batch.permute(0,3,1,2) / 255.0
                next_obs_batch = next_obs_batch.permute(0,3,1,2) / 255.0

            current_q_values = self.current_network(obs_batch).gather(1, act_batch)
            next_q_values = self.target_network(next_obs_batch)
            next_v_values = next_q_values.max(dim = 1)[0].detach()
            target_q_values = rew_batch + self.gamma * next_v_values * (1 - done_mask)

            #CHECKLIST
            #obs_batch the same it looks like
            #act_batch the same it looks like
            #rew_batch the same it looks like
            #next_obs_batch the same it looks like
            #done_mask the same
            #current_q_values the same
            #next_q_values fixed and now same

            #...
            #loss looks to be the same!
            #initial gradients are the same
            #clipped gradients look the same
            #

            loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))

            # if self.t >= 10000:
            #     pdb.set_trace()

            # pdb.set_trace()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.current_network.parameters(), 10)


            # gradients look the same???
            # for p in self.current_network.parameters():
            #     print(p.data)
            #     print(p.grad.data)
            #     print("-------------------------")
            # pdb.set_trace()
            self.optimizer.step()

            #still not the same after
            # print([p for p in self.current_network.parameters()])
            # pdb.set_trace()
            #-0.095480812


            if self.num_param_updates % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.current_network.state_dict())

            self.num_param_updates += 1
        self.t += 1
