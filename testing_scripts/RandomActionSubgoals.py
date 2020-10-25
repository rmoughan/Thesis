import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../envs/')
from montezuma_wrapper import MontezumaWrapper
sys.path.append('../infrastructure/')
from subgoal_buffer import SubgoalBuffer

### Officially marking this as a TODO and moving on for now. Things I need to do:
    #1) Debug -- best way to do it right now is probably do a pointwise debug
    #2) Clean up the shit in SubgoalBuffer

import pdb

env = gym.make('MontezumaRevengeNoFrameskip-v4')
env = MontezumaWrapper(env)

#seed everything
random.seed(42)
np.random.seed(42)
env.seed(42)

subgoal_buffer = SubgoalBuffer()
Goals = []

num_episodes = 350
num_random_actions = 2000

# all_actions = [[np.random.randint(0,8) for _ in range(num_random_actions)] for _ in range(num_episodes)]

finish_room_actions = [5 for _ in range(34)] + [3 for _ in range(45)] + [6] + [0 for _ in range(20)] + \
[6] + [0 for _ in range(40)] + [5 for _ in range(40)] + [4 for _ in range(45)] + [7] + [0 for _ in range(25)] + \
[4 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [1] + [0 for _ in range(25)] + \
[3 for _ in range(4)] + [5 for _ in range(40)] + [3 for _ in range(45)] + [6] + [0 for _ in range(30)] + \
[3 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [7] + [0 for _ in range(20)] + \
[7] + [0 for _ in range(30)] + [4 for _ in range(7)] + [2 for _ in range(40)] + [4 for _ in range(10)] + \
[7] + [0 for _ in range(30)] + [4 for _ in range(35)]

get_to_key_actions = [5 for _ in range(34)] + [3 for _ in range(45)] + [6] + [0 for _ in range(20)] + \
[6] + [0 for _ in range(40)] + [5 for _ in range(40)] + [4 for _ in range(45)] + [7] + [0 for _ in range(25)] + \
[4 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [1] + [0 for _ in range(25)]

top_left = [4,4,4,4,4,7] + [4 for _ in range(50)]

top_right = [3,3,3,3,3,6] + [3 for _ in range(50)]

all_actions = [finish_room_actions] + \
[get_to_key_actions + [np.random.randint(0,8) for _ in range(num_random_actions - len(get_to_key_actions))] for _ in range(49)] + \
[top_left + [np.random.randint(0,8) for _ in range(num_random_actions - len(top_left))] for _ in range(10)] + \
[top_right + [np.random.randint(0,8) for _ in range(num_random_actions - len(top_right))] for _ in range(10)] + \
[[np.random.randint(0,8) for _ in range(num_random_actions)] for _ in range(num_episodes - 70)]

for episode in range(num_episodes):
    observation = env.reset()
    total_reward = 0
    for t in range(num_random_actions):
        action = all_actions[episode][t]
        obs, reward, done, info = env.step(action)
        jumping, inAir, lives, loc = info["jumping"], info["inAir"], info["ale.lives"], info["loc"]
        # if loc == (58.53030303030303, 107.6969696969697) and action == 4:
        #     pdb.set_trace()
        # if episode == 194 and t >= 100: #should be at 1372
        #     #why isn't this being registered over there. Weeeeird, the other code can't get the coordinates and thus never stores it.
        #     #In general a subgoal should never exist inAir.
        #
        #     from PIL import Image
        #     print(action, env.vertical_jump_timer, env.lastInAir, inAir)
        #     Image.fromarray(obs.squeeze()).show()
        #     pdb.set_trace()
        total_reward += reward #need to double check that this still makes sense
        if total_reward == 400:
            print("Congratulations!! You've reached the end of the first room :)")
            break
        if jumping:
            jump_outcome = info["jump_outcome"]
            subgoal_buffer.store(loc, action, jump_outcome)
            if jump_outcome == -1:
                break
        if (not jumping) and (not inAir):
            #need logic to override reward if die by walking, aka skull
            reward = reward if lives == 6 else -1
            subgoal_buffer.store(loc, action, reward)
        if lives == 5:
            #episode ends anytime ALE dies
            break

    print("End of Random Action Sequence {0}".format(episode))
    subgoals = subgoal_buffer.find_subgoals()
    print(len(subgoals))

#One outstanding issue I can envision -- subgoals used to be states, but now they are not.
#This means ALE will not know when to jump because the skull is ahead versus just
#jumping because ALE reached a coordinate that had the skull ahead in a past episode,
#but doesn't currently. A brute force way to resolve this is by checking the location of ALE,
#and if ALE is in the skull zone, storing the state as well. Then we'd have to do
#an additional check in at_subgoal to see first if at coordinate of skull zone, then
#if the state also matches up (and therefore the skull is ahead). I think I'm ok with
#this from a hack perspective, as ALE already has access to the state in general, so
#I don't think we're cheating our algorithm or anything.

print(subgoals)
#plot a heatmap
env.reset()
obs = env.get_frame()
plt.imshow(obs)

for subgoal_location in subgoals:
    plt.scatter(subgoal_location[0], subgoal_location[1], c = "yellow", alpha = 0.5, s = 10)
plt.show()
print("Done")
