from agents.base import BaseAgent
from networks.convNet import ConvNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
from PIL import Image

class DQN(BaseAgent):
    def __init__(self, config, logger, device):
        super().__init__(config, logger, device)
        self.current_network = ConvNet(self.network_architecture_config)
        if self.debugging:
            self.current_network.load_state_dict(torch.load(self.experiment_filepath + "/current_network_weights"))
        self.current_network.to(self.device)
        self.target_network = ConvNet(self.network_architecture_config)
        self.target_network.load_state_dict(self.current_network.state_dict())
        if self.debugging:
            self.current_network.load_state_dict(torch.load(self.experiment_filepath + "/target_network_weights"))
        self.target_network.to(self.device)
        self.optimizer = torch.optim.Adam(self.current_network.parameters(), lr = self.lr._v, eps = 5e-4)
        #eps is weird and need to potentially mess around with built in lr schedule class in pytorch

    def step(self):

        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs, self.env.current_goal)

        # if self.hierarchical:
        #     failure_rate = 1 - (len([i for i in self.logger.goals_reached[-100:] if i == self.env.current_goal]) / 100)
        #     epsilon = self.epsilon.value(self.env.current_goal, failure_rate)
        # else:
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

        video = self.video if self.recording else None
        obs, reward, done, info = self.env.step(action, video)

        #all added just for Montezuma's Revenge. Could wrap in a conditional later. Also thinking about just folding it into the wrapper.
        jumping = info["jumping"]
        if jumping:
            reward = info["jump_outcome"]
            inAir = info["inAir"]
            while jumping or inAir:
                obs, _, done, info = self.env.step(0, video)
                jumping = info["jumping"]
                inAir = info["inAir"]
                self.t += 1
                self.episode_length += 1
                if self.t % self.report_freq == 0:
                    failure_rates = [1 - (len([i for i in self.logger.goals_reached[-100:] if g in i]) / 100) for g in self.goals]
                    # self.logger.report(self.t, [round(self.epsilon.value(self.goals[i], failure_rates[i]), 3) for i in range(len(self.goals))])
                    self.logger.report(self.t, self.epsilon.value(self.t))
            self.t -= 1
            self.episode_length -= 1

        if info["ale.lives"] < 6 or reward < 0:
            self.intrinsic_reward = -1
        elif info["at_goal"] or reward == 1:
            self.intrinsic_reward += 1
            self.goals_reached.append(self.env.current_goal)
        else:
            self.intrinsic_reward = 0

        done = done or self.intrinsic_reward == -1

        self.replay_buffer.store_effect(self.replay_buffer_idx, action, self.intrinsic_reward, done)

        self.episode_reward += reward
        self.at_goal = info["at_goal"] or reward == 1

        self.last_obs = obs
        self.done = done


        ### OVERALL THINGS I NEED TO DO:
        #1) Figure out navigation. See below.
        #2) Figure out graphical representation of space. 

        ### NEXT STEPS
        #1) Test controller on numerous subgoals:
            #vertical jump: (79, 68).Got to 90% accuracy within 75k iterations with the below exploration schedule
            #a bit down ladder (80, 104). Got to 98% accuracy within 400k iterations with below exploration schedule
            #the rope (111, 121). Got to 90% accuracy within 75k iterations with the below exploration schedule
            #right part of top edge (88, 84). Got to 90% within 500k iterations with the below exploration schedule
            #bottom of ladder (78, 127). Got to 90% accuracy consistently within 1.6m iterations with below exploration schedule. Not great, but not terrible. Was at 50% 1m in, 70% 1.2m in. Kinda want to try different schedule

            #Was using epsilon as PiecewiseSchedule([(0,1.00), (1e5,0.10), (5e5, 0.01)], outside_value=0.01)
            #A good 1-2 subgoal sequence to test would be right part of top edge to the top right edge (so like 10 steps right and then jump right)

            #Runtime baseline: 0.045/10k on cpu, 0.018/10k Tesla T4 gpu. Latter still seems a bit high but I'll watch it.

        #2) This looks pretty good. I'd like to potentially customize the epsilon schedule a bit more, but this is promising. I would vote that next
            #write the code to test basic 1-2 subgoal sequences. Probably rope-bottom of ladder and top edge-right edge. This should allow me to write some
            #code where the metacontroller will be in the future. Got to 100%, 97% accuracy within 500k iterations with above exploration schedule
            # - Should probably modify epsilon schedule to reset for every subgoal. Will check paper.

        #3) Debugging ideas for issues with rope-middle_right_platform-bottom_right_ladder:
            # - Go back to what I had and confirm runtime and all of the above.
            # - first one that's obvious ish is to test a different sequence without a ladder and see if it can learn it. Coming up with a good one is
            # a bit hard because the stage is so separated by ladders.
            # - It looks like the binary goal mask may include the image, which could make a difference
            # - More rigorously, develop a testing platform so that I can directly read out the Q values the network is outputting and figure ouot what is
            # actually going on. Then debug. Will probably start on this while I do the above and some hyperparameter testing since it doesn't totally suck. But how...
                # > Design ideas: First thing that would be helpful would be to build in storing/loading weights for nets. I think it makes sense to store these
                # in the experiment folder, that shouldn't make anything too bad. Then shall I load up in the __init__ method? Also should set up a different config
                # just so that the learning_starts would be at infinity and epsilon is at 0, may need to change other things. Then all I would need to do is feed it
                # actions to get it to a specific spot and then I can read out the q values from the network. I think this should work, and isn't tremendous overhead.
                # I may need to also store/load the optimizer params, need to think about that for a sec.


        ### OUTSTANDING CONCERNS THAT I HAVE
        # 1) Jumping... not a great fix here. Kinda just have to test and see what happens. Especially considering the obs encoding is the previous 3 frames?
            #could try hardcoding jump length for just this kind of jump to see if it can learn when it's literally just a jump right then goal
            #can try just feeding in jump rights for every single action. More generally, for closer goals I can feed it the right "random actions"
            #so that in theoory it should be able to update the q_network more often
        # 2) Non-stationarity at the goal. Could remedy this by adding a test to see if ale is moving to the at_goal method, wouldn't be to hard I think. Would cost runtime though
        # 3) In the (likely) event where I have to debug the meta-controller, can start by commenting out controller and replacing with perfectly deterministic paths between gooals
        # 4) Could be nice to have some kind of map of the Value function over the state space
        # 5) Why do we randomly sample when the reward signal is sparse? Wouldn't we want to put emphasis on samples with a nonzero reward?
        # 6) Can I instead learn the simpler representation and do RL on that? (See 8)
        # 7) Boltzmann exploration instead of epsilon greedy exploration?
        # 8) Storing subgoals in a graph
            # - reduces potential combinatorial problem
            # - how to do this in a memory efficient way is a bit interesting.
            # - if I run in to combinatorial problems, can look at neaest k subgoals

    def update(self):

        if (self.t > self.learning_starts and \
            self.t % self.learning_freq == 0 and \
            self.replay_buffer.can_sample(self.batch_size)):


            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)
            #could try moving this above to step for a bit more efficiency
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

            loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.current_network.parameters(), 10)

            self.optimizer.step()

            if self.num_param_updates % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.current_network.state_dict())

            self.num_param_updates += 1

        self.t += 1
