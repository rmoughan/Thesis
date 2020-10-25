import sys
sys.path.append('../envs/')
from montezuma_wrapper import *
import time
import matplotlib.pyplot as plt


###NOTE THAT THIS IS CURRENTLY INCORRECT BECAUSE OF CHANGES TO THE ACTION SPACE. I WILL FIX THIS LATER.

from PIL import Image
def display_image(obs):
    img = Image.fromarray(obs)
    img.show()
    return None


env = gym.make('MontezumaRevengeNoFrameskip-v4')
env = MontezumaWrapper(env)
obs = env.reset()
last_action = 0
#jump right to the rope
# actions = [11] + [0 for _ in range(60)]
#jump to the upper right platform
# actions = [3,3,3,3,3,3,3,3,3,3,3,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#go down the treadmill and walk to the rightmost edge
# actions = [5 for _ in range(34)] + [3 for _ in range(45)]
#go down the treadmill and walk left and die

#jump left and die
# actions = [0,0,0,0,0,0,12] + [0 for _ in range(115)]
#walk left and die
# actions = [4 for _ in range(50)]
#walk right and die
# actions = [3 for _ in range(50)]
#testing the vertical jump on ladder issue
# actions = [5,5,5] + [1,5] * 50
#get to and jump over the skull
# actions = [5 for _ in range(34)] + [3 for _ in range(45)] + [11] + [0 for _ in range(20)] + \
# [11] + [0 for _ in range(40)] + [5 for _ in range(40)] + [4 for _ in range(45)] + [12] + [0 for _ in range(25)]
#get to the key
# actions = [5 for _ in range(34)] + [3 for _ in range(45)] + [11] + [0 for _ in range(20)] + \
# [11] + [0 for _ in range(40)] + [5 for _ in range(40)] + [4 for _ in range(45)] + [12] + [0 for _ in range(25)] + \
# [4 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [1] + [0 for _ in range(25)]
# finish the room
# actions = [5 for _ in range(34)] + [3 for _ in range(45)] + [11] + [0 for _ in range(20)] + \
# [11] + [0 for _ in range(40)] + [5 for _ in range(40)] + [4 for _ in range(45)] + [12] + [0 for _ in range(25)] + \
# [4 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [1] + [0 for _ in range(25)] + \
# [3 for _ in range(4)] + [5 for _ in range(40)] + [3 for _ in range(45)] + [11] + [0 for _ in range(30)] + \
# [3 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [12] + [0 for _ in range(20)] + \
# [12] + [0 for _ in range(30)] + [4 for _ in range(7)] + [2 for _ in range(40)] + [4 for _ in range(10)] + \
# [12] + [0 for _ in range(30)] + [4 for _ in range(35)]
#jump left off treadmill and die
# actions = [5 for _ in range(34)] + [4 for _ in range(3)] + [12] +  [0 for _  in range(50)]
#walk left off treadmill and die
# actions = [5 for _ in range(34)] + [4 for _ in range(50)]
#walk right off treadmill and die
# actions = [5 for _ in range(34)] + [3 for _ in range(80)]
#fall off rope and die
# actions = [11] + [0 for _ in range(60)] + [5 for _ in range(50)]

#could add some fuzz testing but that's also in randomly found subgoals script

locs = []
for t in range(len(actions)):
    env.render()
    action = actions[t]
    obs, reward, done, info = env.step(action)
    if info["ale.lives"] < 6:
        break
    jumping = info["jumping"]

    # USED FOR TESTING LOCATION
    # locs.append(info["loc"])

    if jumping:
        print("Your Airness")
        print(info["jump_outcome"])
        time.sleep(1)
    time.sleep(0.1)

# NOTE TO SELF: FOR NOW THE LOCATION IS IN 210x160 coordinates and I THINK
# that should be fine but I may have to switch back later.

# USED FOR TESTING LOCATION
# env = gym.make('MontezumaRevengeNoFrameskip-v4')
# obs = env.reset()
# plt.imshow(obs)
# for loc in locs:
#     plt.scatter(loc[0], loc[1], c = "yellow", alpha = 0.5, s = 10)
# plt.show()

#Main diff is I don't have inAir removing header, may cause issues. But I think
#it may be more of a feature than a bug? The other thing is the rendering
#artefact from cloning that slightly bothers me.

env.close()
