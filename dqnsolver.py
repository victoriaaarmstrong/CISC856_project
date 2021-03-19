import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

GAMMA = None
LEARNING_RATE = None

BATCH_SIZE = None
MEMORY_SIZE = None

EXPLORATION_MIN = None
EXPLORATION_MAX = None
EXPLORATION_DECAY = None


class DQNSolver(nn.model):

    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).init()

        self.exploration_rate = EXPLORATION_MAX
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.conv = nn.Sequential(
            ## haven't played around with any parameters here - used old ones
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            ## haven't played around with any parameters here - used old ones
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        def _get_conv_out(self, shape):
            o = self.conv(torch.zeros(1, *shape))

            return int(np.prod(o.size()))

        def forwardOne(self, image):
            conv_out = self.conv(image).view(image.size()[0], -1)

            return self.fc(conv_out)

        def forwardTwo(self, image1, image2):
            out_one = self.conv(image1).view(image1.size()[0], -1)
            out_two = self.conv(image2).view(image2.size()[0], -1)

            ## idk pytorch so idk if this is doing what I want it to do
            x = torch.cat((out_one, out_two), dim=1)

            return self.fc(x)


        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def act(self, state):
            """

            we implemented these in different spots/different methods so I wanted to check first
            :param self:
            :param state:
            :return:
            """
            return

        def experience_replay(self):
            """

            we implemented these in different spots/different methods so I wanted to check first
            :param self:
            :return:
            """
            return