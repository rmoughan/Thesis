import random
import numpy as np
from infrastructure.utils import *
import pdb

#Taken and modified from CS 182 who took it from CS 285

class OffPolicyReplayBuffer(object):

    def __init__(self, size, frame_history_len, cartpole, hierarchical):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.cartpole = cartpole
        self.size = size
        self.frame_history_len = frame_history_len
        self.next_idx = 0
        self.num_in_buffer = 0
        self.obs = None
        self.action = None
        self.reward = None
        self.done = None
        self.hierarchical = hierarchical
        if self.hierarchical:
            self.goal = None
            self.encoded_goal_masks = {}

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for iter_idx in range(start_idx, end_idx - 1):
            if self.done[iter_idx % self.size]:
                start_idx = iter_idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            obs_out = np.concatenate(frames, 2)

        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            obs_out = self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1) #can change this to .transpose(3,0,1,2) if I really want to

        if not self.hierarchical:
            return obs_out

        base_img = obs_out[:,:,3]
        goal_coords = tuple(self.goal[idx]) #I feel like it should be looking back (so idx), not looking forward (start_idx)
        if goal_coords in self.encoded_goal_masks.keys():
            goal_out = self.encoded_goal_masks[goal_coords]
        else:
            goal_out = self._encode_goal_state(goal_coords[0], goal_coords[1], base_img)
            self.encoded_goal_masks[goal_coords] = goal_out

        concat_out = np.concatenate((obs_out, goal_out), axis = 2)
        return concat_out

    def store_frame(self, frame, goal = None):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs    = np.empty([self.size] + list(frame.shape), dtype=np.float32 if self.cartpole else np.uint8)
            self.action = np.empty([self.size],                     dtype=np.int32)
            self.reward = np.empty([self.size],                     dtype=np.float32)
            self.done   = np.empty([self.size],                     dtype=np.bool)
            if self.hierarchical:
                self.goal = np.empty((self.size, 2), dtype = np.uint8)

        self.obs[self.next_idx] = frame
        if self.hierarchical:
            self.goal[self.next_idx] = goal

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done


    def _encode_goal_state(self, goal_x, goal_y, base_img):

        m = np.zeros((210,160,3), dtype = np.float32)
        m[goal_y - 4: goal_y + 5, goal_x - 4: goal_x + 5,:] = 255 #255 to offset the div later on, +/- because of the k parameter in the wrapper
        m = convert_image_84(m).squeeze()
        base_img[np.nonzero(m)] = 255 #hacky as hell but does the job for now. Don't care unless there's efficiency issues
        base_img = base_img.reshape(84, 84, 1)

        return base_img
