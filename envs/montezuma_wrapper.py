import numpy as np
import gym
from gym import spaces
import cv2

import pdb

class MontezumaWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype = np.uint8)
        self.action_space = spaces.Discrete(8)
        self.goals = []
        self.current_goal = None
        self.goal_dist_limit = 4 #hyperparameter to be tweaked. Anything less than 4 has issues with the end of the jump not lining up with the goal location
        # self.vertical_jump_timer = 0

    def step(self, action, video = None):

        action = self._convert_action(action)
        obs, reward, done, info = self.env.step(action)
        if video != None:
            video_obs = np.zeros(obs.shape, dtype = np.uint8)
            video_obs[:,:,0] = obs[:,:,2]
            video_obs[:,:,1] = obs[:,:,1]
            video_obs[:,:,2] = obs[:,:,0]
            video.write(video_obs)
        inAir = self.inAir(obs, action, self.last_action)
        jumping = self.lastInAir == False and inAir == True
        if jumping:
            jump_outcome = self.getJumpOutcome(info["ale.lives"])
            info["jump_outcome"] = jump_outcome
        info["inAir"], info["jumping"] = inAir, jumping
        loc = self.getAleLocation(obs)
        info["loc"] = loc
        if self.current_goal != None:
            info["at_goal"] = self.at_goal(loc, self.current_goal)
        self.lastInAir = inAir
        self.last_action = action
        jumping = False

        obs = self.process_frame(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.process_frame(obs)
        self.last_action = 0
        self.lastInAir = False
        return obs

    def _process_frame84(self, frame): #taken from deepmind
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

    def _remove_treadmill(self, frame):
        frame[136:141,60:100,:] = np.full(shape = (5,40,3), fill_value = 127)
        return frame

    def _remove_header(self, frame):
        frame[:48,:,:] = np.zeros((48,160,3))
        return frame

    def _remove_skull(self, frame):
        frame[165:180,54:112,:] = np.zeros((15, 58, 3))
        return frame

    def _remove_key(self, frame):
        frame[98:116,10:23,:] = np.zeros((18, 13, 3))
        return frame

    def process_frame(self, frame):
        #deal with all that sweet sweet Montezuma's Revenge bullshit here
        frame = self._remove_treadmill(frame)
        frame = self._remove_header(frame)
        frame = self._process_frame84(frame)
        return frame

    def inAir(self, original_obs, action, last_action):

        # if action == 1 and self.lastInAir == False:
        #     self.vertical_jump_timer = 20
        #
        # if self.vertical_jump_timer > 0:
        #     self.vertical_jump_timer -= 1
        #     return True

        test_action = 0
        clone_state = self.env.clone_full_state()
        for _ in range(2):
            obs, reward, done, info = self.env.step(test_action)
            test_obs, reward, done, info = self.env.step(test_action)
        self.env.restore_full_state(clone_state)

        obs = self._remove_key(self._remove_skull(self._remove_treadmill(obs)))
        test_obs = self._remove_key(self._remove_skull(self._remove_treadmill(test_obs)))

        key_test_obs = test_obs[23:45,53:67,:]
        key_original_obs = original_obs[23:45,53:67,:]

        if np.any(key_test_obs - key_original_obs):
            return True

        if not np.any(test_obs - obs):
            return False

        #dealing with the annoying case where on treadmill and always moving
        treadmill_observation = original_obs[135:136,63:100,:]
        # treadmill_test_obs = test_obs[135:136,63:100,:] #for when I come back to dealing with the environment
        # if np.any(treadmill_test_obs):
        #     return False
        valid_jumps = [1,6,7]
        if np.any(treadmill_observation) and action not in valid_jumps and last_action not in valid_jumps:
            return False

        return True


    def getJumpOutcome(self, original_lives):
        #outcomes: death (-1), no death (0), reward (1). Changed to be the actual rewards for now, both extrinsic and intrinsic
        action = 0
        clone_state = self.env.clone_full_state()
        # stored_jump_timer = self.vertical_jump_timer
        obs, reward, done, info = self.env.step(action)
        while True:
            obs, reward, done, info = self.env.step(action)
            at_goal = False
            if self.current_goal != None:
                loc = self.getAleLocation(obs)
                at_goal = self.at_goal(loc, self.current_goal)
            inAir = self.inAir(obs, action, action)
            lives = info['ale.lives']
            if reward > 0 or at_goal:
                self.env.restore_full_state(clone_state)
                # self.vertical_jump_timer = stored_jump_timer
                return int(reward or at_goal)
            if lives < original_lives:
                self.env.restore_full_state(clone_state)
                # self.vertical_jump_timer = stored_jump_timer
                return -1
            if not inAir:
                obs, reward, done, info = self.env.step(action)
                lives = info['ale.lives']
                if lives < original_lives:
                    self.env.restore_full_state(clone_state)
                    # self.vertical_jump_timer = stored_jump_timer
                    return -1
                self.env.restore_full_state(clone_state)
                # self.vertical_jump_timer = stored_jump_timer
                return 0

    def getAleLocation(self, frame):
        obs = frame
        for test_action in [4,5,6]:
            diff = self._attempt_action(obs, test_action)
            if np.sum(diff) > 0:
                break
        if np.sum(diff) == 0:
            raise ValueError("Not able to determine location!")
        nonzero_coords = np.where(diff[:,:,0] != 0)
        [mean_y, mean_x] = [np.mean(nonzero_coords[0]),np.mean(nonzero_coords[1])]
        coords = (float(mean_x),float(mean_y))
        return coords

    def _attempt_action(self, obs, test_action):
        clone_state = self.env.clone_full_state()
        obs = self._remove_key(self._remove_skull(self._remove_header(self._remove_treadmill(obs))))
        for i in range(5):
            next_obs, _, _, _ = self.env.step(test_action)
            next_obs = self._remove_key(self._remove_skull(self._remove_header(self._remove_treadmill(next_obs))))
            diff = next_obs - obs
            if np.sum(diff) != 0:
                self.env.restore_full_state(clone_state)
                return diff
            obs = next_obs
        self.env.restore_full_state(clone_state)
        return np.zeros(1)

    def at_goal(self, loc, goal):
        dist = np.linalg.norm(np.array(goal) - np.array(loc))
        if dist <= self.goal_dist_limit:
            return True
        return False

    def add_goal(self, goal_coords):
        self.goals += [goal_coords]

    def _convert_action(self, action):
        if action >= 0 and action <= 5:
            return action
        elif action == 6:
            return 11
        elif action == 7:
            return 12
        else:
            raise ValueError("Invalid action")

    def get_frame(self):
        clone_state = self.env.clone_full_state()
        obs, _, _, _, = self.env.step(0)
        self.env.restore_full_state(clone_state)
        return obs



### MAP OF ALL ACTIONS AND THEIR MEANING:
#0: Noop
#1: Vertical Jump
#2: Step Up
#3: Step Right
#4: Step Left
#5: Step Down
#6: Step Right
#7: Step Left
#8: Step Right
#9: Step Left
#10: Vertical Jump
#11: Right Jump
#12: Left Jump
#13: Step Down
#14: Right Jump
#15: Left Jump
#16: Right Jump
#17: Left Jump

### UNIQUE ACTIONS:
#0: Noop
#1: Vertical Jump
#2: Step Up
#3: Step Right
#4: Step Left
#5 Step Down
#6: Right Jump
#7: Left Jump

# get_to_skull_actions = [5 for _ in range(34)] + [3 for _ in range(45)] + [6] + [0 for _ in range(20)] + \
# [6] + [0 for _ in range(40)] + [5 for _ in range(40)] + [4 for _ in range(30)]
#
# from PIL import Image
# import pdb
# import time
# env = gym.make("MontezumaRevengeNoFrameskip-v4")
# env = MontezumaWrapper(env)
# env.reset()
# # for i in range(len(get_to_skull_actions)):
# #     env.step(get_to_skull_actions[i], 0, 0)
#
# env.step(1, 0, 0)
# for i in range(33):
#     obs, action, reward, info = env.step(0, 0, 0)
#     print(i, env.vertical_jump_timer, info["inAir"], info["loc"])
#     Image.fromarray(env.get_frame()).show()
#     pdb.set_trace()
