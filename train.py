import sys
from infrastructure.config import Config
from infrastructure.logger import Logger
# from agents.dqn import DQN
from agents.dqn_copy import DQN #change back later
from agents.actor_critic import Actor_Critic
import os
import torch

import pdb

if len(sys.argv) != 2:
    raise ValueError("Invalid command. Try running 'python train.py <config_file>'")
config_file = sys.argv[1]
config = Config(config_file)

logger = Logger(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Device: CPU")

model = config.params["model"]
model = eval(model)(config, logger, device)

# import pickle
# import torch
# import numpy as np
#
# infile = open("./debugging/actor_initializations", "rb")
# initial_weights = pickle.load(infile)
# infile.close()
#
# i = 0
# for p in model.actor.parameters():
#     if len(initial_weights[i].shape) == 2:
#         initial_weights[i] = initial_weights[i].T
#     p.data = torch.from_numpy(initial_weights[i])
#     i += 1
#
# infile = open("./debugging/critic_initializations", "rb")
# initial_weights = pickle.load(infile)
# infile.close()
#
# i = 0
# for p in model.critic.parameters():
#     if len(initial_weights[i].shape) == 2:
#         initial_weights[i] = initial_weights[i].T
#     p.data = torch.from_numpy(initial_weights[i])
#     i += 1
#
# #need to re-move the model, ugh
# model.actor.to(model.device)
# model.critic.to(model.device)

# import pickle
# import torch
# import numpy as np
#
# infile = open("./debugging/convnet_initializations", "rb")
# initial_conv_weights = pickle.load(infile)
# infile.close()
#
# i = 0
# for p in model.current_network.parameters():
#     if len(initial_conv_weights[i].shape) == 4:
#         initial_conv_weights[i] = np.transpose(initial_conv_weights[i], (3,2,0,1))
#     elif len(initial_conv_weights[i].shape) == 2:
#         initial_conv_weights[i] = np.transpose(initial_conv_weights[i], (1,0))
#     p.data = torch.from_numpy(initial_conv_weights[i])
#     i += 1
#
# infile = open("./debugging/target_convnet_initializations", "rb")
# target_initial_conv_weights = pickle.load(infile)
# infile.close()
#
# i = 0
# for p in model.target_network.parameters():
#     if len(target_initial_conv_weights[i].shape) == 4:
#         target_initial_conv_weights[i] = np.transpose(target_initial_conv_weights[i], (3,2,0,1))
#     elif len(target_initial_conv_weights[i].shape) == 2:
#         target_initial_conv_weights[i] = np.transpose(target_initial_conv_weights[i], (1,0))
#     p.data = torch.from_numpy(target_initial_conv_weights[i])
#     i += 1
#
# #need to re-move the model, ugh
# model.current_network.to(model.device)
# model.target_network.to(model.device)

# pdb.set_trace()

# for p in model.current_network.parameters():
#     print(p)

# import pickle
# import torch
# infile = open("./debugging/net_initializations", "rb")
# initial_weights = pickle.load(infile)
# infile.close()
#
# infile = open("./debugging/target_net_initializations", "rb")
# target_initial_weights = pickle.load(infile)
# infile.close()



# i = 0
# for p in model.current_network.parameters():
#     # print(p)
#     # print(p.data.shape) #set in here with p.data.fill_(value)
#     if len(initial_weights[i].shape) == 2:
#         initial_weights[i] = initial_weights[i].T
#     # print(initial_weights[i].shape)
#     p.data = torch.from_numpy(initial_weights[i])
#     # print("---------------------")
#     i += 1
#
# i = 0
# for p in model.target_network.parameters():
#     # print(p)
#     # print(p.data.shape) #set in here with p.data.fill_(value)
#     if len(target_initial_weights[i].shape) == 2:
#         target_initial_weights[i] = target_initial_weights[i].T
#     # print(initial_weights[i].shape)
#     p.data = torch.from_numpy(target_initial_weights[i])
#     # print("---------------------")
#     i += 1

# for p in model.current_network.parameters():
#     print(p)

#looks like I can potentially set the weight initializations to be a copy
#of the 182 ones if necessary
#Current candidates that are definitely different: OptimizerSpec

def main():
    print("Hello and welcome to my thesis...")
    model.train()

if __name__ == "__main__":
    main()

#current idea: implement DQN, PPO, PPOC, make all hRL

#TODO: Add support for other activations
