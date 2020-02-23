

from gym import spaces, Env
import gym
from src.utils import *
from src.utils_plot import *
import random

import os


import numpy as np
import matplotlib.pyplot as plt

MAX_STEPS = 1000
MAX_VALUES = 1000
MAX_LOSS = 10000




class WhirlwindEnvironment(Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config_dict, config_input_data):
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
        self.betta = config_input_data['betta_0']
        if not self.config_dict['use_the_random_noise']:
            self.alfa = config_input_data['alfa_0']
        else:
            self.alfa = 0

        # self.iteration = config_dict['iteration']


        if self.config_dict['use_the_random_noise']:
            self.ksi = generate_noise_dots(self.n_whirlwind, config_input_data['distribution_params'])
        else:
            self.ksi = np.zeros(self.points_z.shape)

        self.action_space = spaces.Discrete(3) #spaces.Box(
            #low=0, high=1, shape=(1, ), dtype=np.int)

        # Prices contains the OHCL values for the last five prices
        # + betta + distance change
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.points_z.size + 1 + 3,), dtype=np.float16)
            #spaces.Box(low=0, high=np.inf, shape=self.points_z.shape, dtype=np.float16),


        self.force_mag = 0.001

        self.II_pev = circulation_moment(self.points_z)
        self.current_betta = self.betta
        self.current_alfa = self.alfa
        self.current_position = self.points_z
        self.betta_list = []
        self.loss = self.II_pev
        self.current_step = 0





    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        #print(" current_position", self.current_position)
        new_position = next_z_generate_point(self.current_position,
                                             self.ksi,
                                             self.current_betta,
                                             self.current_alfa,
                                             self.config_dict['use_second_interaction'])

        #print("new_position", new_position)
        #print("[self.current_betta]", [self.current_betta])
        distance = [calculate_the_measure(new_p, old_p)  for (new_p, old_p) in zip(new_position,self.current_position)]
        obs =  np.concatenate(new_position)
        obs =  np.concatenate([obs, [self.current_betta], distance])
        self.current_position = np.stack(new_position, axis=0)
        # print("next _obs self.current_position", self.current_position)
        #print("obs", obs)
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        if action == 0:
            pass
        elif action == 1:
            self.current_betta += self.force_mag

        elif action == 2:
            new_betta = self.current_betta - self.force_mag
            # i = 1
            # while new_betta < 0:
            #     new_betta = self.current_betta - self.force_mag / i
            #     i += 1

            self.current_betta = new_betta

        # self.current_betta += self.force_mag if action == 1 else -self.force_mag

    def info_dictionary(self, action):

        main_measure_dict = {
            'action': action,
            'points': self.current_position,
            'center': circulation_center(self.current_position),
            'moment': circulation_moment(self.current_position),
            'dispersion': circulation_dispersion(self.current_position),
            'alfa': self.alfa,
            'betta': self.betta,
            #'gamma': self.gamma,
            'alfa_loss': self.loss
              }
        return main_measure_dict

    def step(self, action):

        self.betta_list.append(self.current_betta )
        print("betta_list: ", self.betta_list)
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        if self.current_betta <= 0 :
            obs = self._next_observation()
            done = True
            reward = - 100


        else:
            print("action: ", action)
            II_next = circulation_moment(self.current_position)

            self.loss = np.abs(self.II_pev - (II_next + np.sum([
                save_div(1 - betta, betta)  for betta in self.betta_list
                            ])))
            self.II_pev = II_next

            reward = - self.loss

            if self.loss >= 10000:

                obs = self._next_observation()
                done = True
                reward = - 100


            else:
                print("reward: ", reward)
                done = self.current_step >= MAX_STEPS
                #print("done: ", done)
                obs = self._next_observation()

                # print("step :", self.current_position)
                #print(f'Loss : {self.loss}')


        return obs, reward, done, self.info_dictionary(action)

    # 3 done - stop
    # 4 info debug {}


    def reset(self):
        # Reset the state of the environment to an initial state

        self.current_betta = self.betta
        self.current_alfa = self.alfa
        self.current_position = self.points_z
        self.betta_list = []
        self.loss = self.II_pev
        self.current_step = 0


        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen

        print(f'Step: {self.current_position}')
        print(f'Reward: {-self.loss}')
        print(f'Betta list: {self.betta_list}')


config_dict = {
    'find_use_best_alfa': False,
    'use_second_interaction': False,
    'use_the_random_noise': False
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


import tensorflow as tf

print(tf.__version__)


from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


import pandas as pd


# It will check your custom environment and output additional warnings if needed
#check_env(env)
# The algorithms require a vectorized environment to run
#env = DummyVecEnv([lambda: WhirlwindEnvironment(config_dict, config_input_data)])


from stable_baselines.common.env_checker import check_env

env = WhirlwindEnvironment(config_dict, config_input_data)
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)


obs = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())


### Test
GO_LEFT = 0
# Hardcoded best agent: always go left!
n_steps = 20
for step in range(n_steps):
  print("Step {}".format(step + 1))
  obs, reward, done, info = env.step(GO_LEFT)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render()
  if done:
    print("Goal reached!", "reward=", reward)
    break



model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)


from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

# Evaluate the agent
mean_reward, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes=100)



# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))

# Evaluate the agent
mean_reward, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes=100)




from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines import results_plotter

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
#env = gym.make('LunarLanderContinuous-v2')


env = DummyVecEnv([lambda: WhirlwindEnvironment(config_dict, config_input_data)])

env = Monitor(env, log_dir, allow_early_resets=True)

# Add some param noise for exploration
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# Because we use parameter noise, we should use a MlpPolicy with layer normalization
model = DDPG(LnMlpPolicy, env, param_noise=param_noise, verbose=0)
# Train the agent
time_steps = 1e5
model.learn(total_timesteps=int(time_steps), callback=callback)

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG LunarLander")
plt.show()

from stable_baselines import ACKTR
model = ACKTR(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)


obs = env.reset()
aggregate_info = []
for i in range(200):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    aggregate_info.append(info)



main_measure_dict = {k: [dic[k] for dic in  aggregate_info] for k in aggregate_info[0]}


for type_graph in list(main_measure_dict.keys())[2:]:
    try:
        plot_graph = main_measure_dict[type_graph]
        if plot_graph != []:
            plot_metrics(plot_graph, type_graph)
    except ValueError:
        continue


len(main_measure_dict['points'])

sample_points = [main_measure_dict['points'][i] for i in range(0,200)]
plot_the_whirlwind(sample_points)