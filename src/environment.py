from gym import spaces
import gym
from utils import *
import random
import numpy as np

MAX_STEPS = 1000
MAX_VALUES = 1000

class WhirlwindEnvironment(gym.Env):

    def _init_(self, betta_0, alfa_0, config_dict, config_input_data):
        """

        :param points:
        :param betta_0:
        :param alfa_0:
        :param ksi:
        :param config_dict:
        :return:
        """
        super(WhirlwindEnvironment, self).__init__()

        self.n_whirlwind = config_input_data['n_whirlwind']
        self.distribution_params = config_input_data['distribution_params']
        self.points_z = generate_random_dots(
            config_input_data['min_value'],
            config_input_data['max_value'],
            self.n_whirlwind)
        self.config_dict = config_dict
        self.betta = betta_0
        if not self.config_dict['use_the_random_noise']:
            self.alfa = alfa_0
        else:
            self.alfa = 0

        # self.iteration = config_dict['iteration']


        if self.config_dict['use_the_random_noise']:
            self.ksi = generate_noise_dots(self.n_whirlwind, distribution_params)
        else:
            self.ksi = np.zeros(self.points_z.shape)

        self.action_space = spaces.Discrete(2) #spaces.Box(
            #low=0, high=1, shape=(1, ), dtype=np.int)

        # Prices contains the OHCL values for the last five prices
        low_ = np.array([ config_input_data['min_value']  for i in range(len(self.points_z.shape)) ]).reshape(self.points_z.shape)
        high_ = np.array([ MAX_VALUES  for i in range(len(self.points_z.shape)) ]).reshape(self.points_z.shape)
        self.observation_space = spaces.Box(
            low=low_, high=high_, shape=(self.points_z.shape), dtype=np.float16)

        self.force_mag = 0.01

        self.II_null = circulation_moment(self.points_z)
        self.current_betta = self.betta
        self.current_alfa = self.alfa
        self.current_position = self.points_z
        self.betta_list = []
        self.loss = self.II_null

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1

        new_position = next_z_generate_point(self.current_position,
                                             self.ksi,
                                             self.current_betta,
                                             self.current_alfa,
                                             self.config_dict['use_second_interaction'])

        return new_position

    def _take_action(self, action):
        # Set the current price to a random price within the time step


        self.current_betta += self.force_mag if action == 1 else -self.force_mag

        # self.current_betta += self.force_mag if action == 1 else -self.force_mag


    def step(self, action):

        self.betta_list.append(self.current_betta )
        # Execute one time step within the environment
        self._take_action(action)

        self.current_position = self._next_observation()

        II_next = circulation_moment(self.current_position)

        self.loss = II_next - self.II_null  + np.sum(np.divide(1, self.betta_list))
        reward = - self.loss

        done = self.improve_loos <= 0

        return self.current_position, reward, done, {}


    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_betta = self.betta
        self.current_alfa = self.alfa
        self.current_position = self.points_z
        self.betta_list = []



        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen

        print(f'Step: {self.current_position}')
        print(f'Balance: {self.loss}')


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


#points = generate_random_dots(config_input_data['min_value'],
#                              config_input_data['max_value'],
#                              config_input_data['n_whirlwind'])

#ksi = generate_noise_dots(config_input_data['n_whirlwind'],
#                          config_input_data['distribution_params'])

#def _main_():

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()



