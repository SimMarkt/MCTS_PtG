# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Power-to-Gas Dispatch Optimization using Monte Carlo Tree Search (MCTS)
# GitHub Repository: https://github.com/SimMarkt/MCTS_PtG
#
# mcts_main:
# > Main script for the PtG-CH4 dispatch optimization.
# > Adapts to different computational environments: a local personal computer ('pc') or a computing cluster with SLURM management ('slurm').
# ----------------------------------------------------------------------------------------------------------------

# --------------------------------------------Import Python libraries---------------------------------------------
import os
import torch as th

# Library for the RL environment
from gymnasium.envs.registration import registry, register
import gymnasium as gym

# Libraries with utility functions and classes
from src.rl_utils import load_data, initial_print, config_print, Preprocessing, Postprocessing
from src.rl_config_mcts import AgentConfiguration
from src.rl_config_env import EnvConfiguration
from src.rl_config_train import TrainConfiguration

import math
import random
from typing import Any, List
from tqdm import tqdm
import numpy as np
import copy

import matplotlib.pyplot as plt

import argparse

class MCTSNode:
    def __init__(self, env, parent=None, action=None, done=False, remaining_steps=72):
        self.env = env  # deepcopy of the env
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.done = done  # whether this state is terminal
        self.remaining_steps = remaining_steps  # Number of steps remaining for the simulation

    def is_terminal(self):
        return self.done

    def is_fully_expanded(self):
        legal_actions = list(range(self.env.action_space.n))
        tried_actions = [child.action for child in self.children]
        return set(tried_actions) == set(legal_actions)

    def best_child(self, c_param=1.41):
        choices_weights = [
            (child.total_reward / child.visits) + 
            c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def most_visited_child(self):
        return max(self.children, key=lambda child: child.visits)
     
class MCTS:
    def __init__(self, iterations=1000, exploration_weight=1.41):
        self.iterations = iterations
        self.exploration_weight = exploration_weight

    def search(self, root_env, rootnodeid):
        root_env_copy = copy.deepcopy(root_env)
        root_node = MCTSNode(root_env_copy)

        for _ in range(self.iterations):
            node = self._select(root_node)
            if not node.is_terminal():
                node = self._expand(node)
            reward = self._simulate(node.env, node.done, node.action, node.remaining_steps)
            self._backpropagate(node, reward)

        return root_node.most_visited_child().action

    def _select(self, node):
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node):
        legal_actions = list(range(node.env.action_space.n))
        tried_actions = [child.action for child in node.children]
        untried_actions = [a for a in legal_actions if a not in tried_actions]

        action = random.choice(untried_actions)
        new_env = copy.deepcopy(node.env)
        obs, reward, terminated, truncated, info = new_env.step(action)
        done = terminated or truncated
        remaining_steps = node.remaining_steps - 1  # Decrease remaining steps by 1 each time we expand (To plan for the 12 h Day-Ahead period)
        child_node = MCTSNode(new_env, parent=node, action=action, done=done, remaining_steps=remaining_steps)
        node.children.append(child_node)
        return child_node

    def _simulate(self, env, done, action, remaining_steps):
        # In the simulation, always take the same action as the expanded node
        sim_env = copy.deepcopy(env)
        total_reward = 0

        step_count = 0
        while not done and step_count < remaining_steps:
            # Perform the same action as the expanded node during rollout
            _, reward, terminated, truncated, _ = sim_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        return total_reward

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent



def computational_resources(TrainConfig):
    """
        Configures computational resources and sets the random seed for the current thread
        :param TrainConfig: Training configuration (class object)
    """
    print("Set computational resources...")
    TrainConfig.path = os.path.dirname(__file__)
    if TrainConfig.com_conf == 'pc': 
        print("---Computation on local resources")
        TrainConfig.seed_train = TrainConfig.r_seed_train[0]
        TrainConfig.seed_test = TrainConfig.r_seed_test[0]
    else: 
        print("---SLURM Task ID:", os.environ['SLURM_PROCID'])
        TrainConfig.slurm_id = int(os.environ['SLURM_PROCID'])         # Thread ID of the specific SLURM process in parallel computing on a computing cluster
        assert TrainConfig.slurm_id <= len(TrainConfig.r_seed_train), f"No. of SLURM threads exceeds the No. of specified random seeds ({len(TrainConfig.r_seed_train)}) - please add additional seed values to RL_PtG/config/config_train.yaml -> r_seed_train & r_seed_test"
        TrainConfig.seed_train = TrainConfig.r_seed_train[TrainConfig.slurm_id]
        TrainConfig.seed_test = TrainConfig.r_seed_test[TrainConfig.slurm_id]
    if TrainConfig.device == 'cpu':    print("---Utilization of CPU\n")
    elif TrainConfig.device == 'auto': print("---Automatic hardware utilization (GPU, if possible)\n")
    else:                       print("---CUDA available:", th.cuda.is_available(), "GPU device:", th.cuda.get_device_name(0), "\n")

def check_env(env_id):
    """
        Registers the Gymnasium environment if it is not already in the registry
        :param env_id: Unique identifier for the environment
    """
    if env_id not in registry:      # Check if the environment is already registered
        try:
            # Import the ptg_gym_env environment
            from env.ptg_gym_env import PTGEnv

            # Register the environment
            register(
                id=env_id,
                entry_point="env.ptg_gym_env:PTGEnv",  # Path to the environment class
            )
            print(f"---Environment '{env_id}' registered successfully!\n")
        except ImportError as e:
            print(f"Error importing the environment module: {e}")
        except Exception as e:
            print(f"Error registering the environment: {e}")
    else:
        print(f"---Environment '{env_id}' is already registered.\n")

def main():
    # --------------------------------------Initialize the RL configuration---------------------------------------
    initial_print()
    MCTSConfig = AgentConfiguration()
    EnvConfig = EnvConfiguration()

    iterations=50
    EnvConfig.scenario = 3

    TrainConfig = TrainConfiguration()
    computational_resources(TrainConfig)
    str_id = config_print(MCTSConfig, EnvConfig, TrainConfig)
    
    # -----------------------------------------------Preprocessing------------------------------------------------
    print("Preprocessing...")
    dict_price_data, dict_op_data = load_data(EnvConfig, TrainConfig)

    # Initialize preprocessing with calculation of potential rewards and load identifiers
    Preprocess = Preprocessing(dict_price_data, dict_op_data, MCTSConfig, EnvConfig, TrainConfig)    
    # Create dictionaries for kwargs of training and test environments
    env_kwargs_data = {'env_kwargs_train': Preprocess.dict_env_kwargs("train"),
                       'env_kwargs_val': Preprocess.dict_env_kwargs("val"),
                       'env_kwargs_test': Preprocess.dict_env_kwargs("test"),}

    # Instantiate the vectorized environments
    print("Load environment...")
    env_id = 'PtGEnv-v0'
    check_env(env_id)                                                                                                   # Check the Gymnasium environment registry
    # env_train, env_test_post, eval_callback_val, eval_callback_test = create_vec_envs(env_id, str_id, MCTSConfig, TrainConfig, env_kwargs_data)          # Create vectorized environments
        
    # ----------------------------------------------MCTS Validation-----------------------------------------------
    print("Run MCTS on the validation set... >>>", str_id, "<<< \n")
    env_test_post = gym.make(env_id, dict_input = env_kwargs_data['env_kwargs_test'], train_or_eval = "eval")

    mcts = MCTS(exploration_weight=1.41, iterations=iterations)

    obs = env_test_post.reset()
    timesteps = Preprocess.eps_sim_steps_test
    stats = np.zeros((timesteps, len(EnvConfig.stats_names)))
    stats_dict_test = {}

    for i in tqdm(range(timesteps), desc='---Apply MCTS planning on the test environment:'):

        action = mcts.search(env_test_post,i)  # Perform MCTS search to get the best action
        obs, _ , _ , terminated, info = env_test_post.step(action)
        print(f' Pot_Rew {info["Pot_Reward"]}, Load_Id {info["Part_Full"]}, Meth_State {info["Meth_State"]}, Rew {info["reward [ct]"]}, Action {action}')

        # Store data in stats
        if not terminated:
            j = 0
            for val in info:
                if j < 24:
                    if val == 'Meth_Action':
                        if info[val] == 'standby':
                            stats[i, j] = 0
                        elif info[val] == 'cooldown':
                            stats[i, j] = 1
                        elif info[val] == 'startup':
                            stats[i, j] = 2
                        elif info[val] == 'partial_load':
                            stats[i, j] = 3
                        else:
                            stats[i, j] = 4
                    else:
                        stats[i, j] = info[val]
                j += 1
        
    for m in range(len(EnvConfig.stats_names)):
        stats_dict_test[EnvConfig.stats_names[m]] = stats[:(timesteps), m]



    print("...finished MCTS validation\n")
    
    # # -----------------------------------------------MCTS Testing-------------------------------------------------
    # print("Postprocessing...")
    # PostProcess = Postprocessing(str_id, MCTSConfig, EnvConfig, TrainConfig, env_test_post, Preprocess)
    # PostProcess.test_performance()
    # PostProcess.plot_results()

    plot_results(stats_dict_test, EnvConfig, iterations)

def plot_results(stats_dict_test, EnvConfig, iterations):
        """Generates a multi-subplot plot displaying time-series data and methanation operations based on the agent's actions"""
        print("---Plot and save RL performance on the test set under ./plots/ ...\n") 

        stats_dict = stats_dict_test
        time_sim = stats_dict['steps_stats'] * EnvConfig.sim_step
        time_sim *= 1 / 3600 / 24   # Converts the simulation time into days
        time_sim = time_sim[:-6]    # ptg_gym_env.py curtails an episode by 6 time steps to ensure a data overhead
        meth_state = stats_dict['Meth_State_stats'][:-6]+1

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=False)
        axs[0].plot(time_sim, stats_dict['el_price_stats'][:-6], label='El.')
        axs[0].plot(time_sim, stats_dict['gas_price_stats'][:-6], 'g', label='(S)NG')
        axs[0].set_ylabel('El. & (S)NG prices\n [ct/kWh]')
        axs[0].legend(loc="upper left", fontsize='small')
        axs0_1 = axs[0].twinx()
        axs0_1.plot(time_sim, stats_dict['eua_price_stats'][:-6], 'k', label='EUA')
        axs0_1.set_ylabel('EUA prices [€/t$_{CO2}$]')
        axs0_1.legend(loc="upper right", fontsize='small')

        axs[1].plot(time_sim, meth_state, 'b', label='state')
        axs[1].set_yticks([1,2,3,4,5])
        axs[1].set_yticklabels(['Standby', 'Cooldown/Off', 'Startup', 'Partial Load', 'Full Load'])
        axs[1].set_ylabel(' ')
        axs[1].legend(loc="upper left", fontsize='small')
        axs[1].grid(axis='y', linestyle='dashed')
        axs1_1 = axs[1].twinx()
        axs1_1.plot(time_sim, stats_dict['Meth_CH4_flow_stats'][:-6]*1000, 'yellowgreen', label='CH$_4$')
        axs1_1.set_ylabel('CH$_4$ flow rate\n [mmol/s]')
        axs1_1.legend(loc="upper right", fontsize='small')  
        
        axs[2].plot(time_sim, stats_dict['Meth_reward_stats'][:-6]/100, 'g', label='Reward')
        axs[2].set_ylabel('Reward [€]')
        axs[2].set_xlabel('Time [d]')
        axs[2].legend(loc="upper left", fontsize='small')
        axs2_1 = axs[2].twinx()
        axs2_1.plot(time_sim, stats_dict['Meth_cum_reward_stats'][:-6]/100, 'k', label='Cum. Reward')
        axs2_1.set_ylabel('Cumulative \n reward [€]')
        axs2_1.legend(loc="upper right", fontsize='small')

        fig.suptitle(f"MCTS_S{EnvConfig.scenario}_Iter{iterations} \n Rew: {np.round(stats_dict['Meth_cum_reward_stats'][-7]/100, 0)} €", fontsize=9)
        plt.savefig(f'plots/MCTS_S{EnvConfig.scenario}_Iter{iterations}_plot.png')

        plt.close()

if __name__ == '__main__':
    main()



