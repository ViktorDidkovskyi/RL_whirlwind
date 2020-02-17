
import gym
from gym import spaces
import pandas as pd

from utils import *


class WhirlwindEnvironment(gym.Env):

    def _init_(self, points, betta_0, alfa_0, ksi, config_dict):
        """

        :param points:
        :param betta_0:
        :param alfa_0:
        :param ksi:
        :param config_dict:
        :return:
        """
        super(WhirlwindEnvironment, self).__init__()
        self.points_z = points
        self.config_dict = config_dict
        self.betta = betta_0
        if not self.config_dict['use_the_random_noise']:
            self.alfa = alfa_0
        else:
            self.alfa = 0

        #self.iteration = config_dict['iteration']

        if self.config_dict['use_the_random_noise']:
            self.ksi = ksi
        else:
            self.ksi = np.zeros(self.points_z.shape)


    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1

        new_position = next_z_generate_point(self.points_z,
                                             self.ksi,
                                             self.betta,
                                             self.alfa,
                                             self.config_dict['use_second_interaction'])

        return new_position

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        pass

    def step(self, action):
        # Execute one time step within the environment
        pass

    def reset(self):
        # Reset the state of the environment to an initial state
        pass

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass




config_dict = {
    'find_use_best_alfa': False,
    'use_second_interaction': False,
    'use_the_random_noise': True
}

config_input_data = {
    'min_value': 0,
    'max_value': 10,
    'n_whirlwind': 3,
    'distribution_params': (0, 0.1),
    'alfa_0': 0,
    'betta_0': 0.01

}


points = generate_random_dots(config_input_data['min_value'],
                              config_input_data['distribution_params'],
                              config_input_data['n_whirlwind'])

ksi = generate_noise_dots(config_input_data['n_whirlwind'],
                          config_input_data['distribution_params'])

#def _main_():

