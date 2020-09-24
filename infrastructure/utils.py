import numpy as np
import random
import torch
from PIL import Image
import cv2

def display_image(obs):
    img = Image.fromarray(obs)
    img.show()
    return None

def convert_image_84(frame): #taken from deepmind
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

def sow_field(seed, env):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    return None

def conv2d_output_dimension(dimension_in, kernel_size = 5, stride = 2):
    return (dimension_in - (kernel_size - 1) - 1) // stride  + 1

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns comparable
    objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res
