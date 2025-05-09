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
from src.ptg_utils import load_data, initial_print, config_print, Preprocessing, plot_results
from src.ptg_config_mcts import MCTS
from src.ptg_config_env import EnvConfiguration

import math
import random
from typing import Any, List
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

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
    EnvConfig = EnvConfiguration()
    mcts = MCTS()
    EnvConfig.path = os.path.dirname(__file__)
    mcts.path = os.path.dirname(__file__)

    str_id = config_print(mcts, EnvConfig)
    
    # -----------------------------------------------Preprocessing------------------------------------------------
    print("Preprocessing...")
    dict_price_data, dict_op_data = load_data(EnvConfig)

    # Initialize preprocessing with calculation of potential rewards and load identifiers
    Preprocess = Preprocessing(dict_price_data, dict_op_data, EnvConfig)    
    # Create dictionaries for kwargs of training and test environments

    # Instantiate the environment
    print("Load environment...")
    env_id = 'PtGEnv-v0'
    check_env(env_id)              # Check the Gymnasium environment registry
    env_test_post = gym.make(env_id, dict_input = Preprocess.dict_env_kwargs())
       
    # ----------------------------------------------MCTS Validation-----------------------------------------------
    print("Run MCTS on the validation set... >>>", str_id, "<<< \n")

    obs, _ = env_test_post.reset()
    timesteps = Preprocess.eps_sim_steps_test
    stats = np.zeros((timesteps, len(EnvConfig.stats_names)))
    stats_dict_test = {}
    pot_reward = 0

    timesteps = 100
    store_interval = 50

    for i in tqdm(range(timesteps), desc='---Apply MCTS planning on the test environment:'):
        
        if i % store_interval == 0 and i != 0: mcts.store_tree()  # Store the tree structure every store_interval steps

        action = mcts.search(env_test_post, Meth_State=obs['METH_STATUS'], init_el_price=obs['Elec_Price'][0], init_pot_reward=pot_reward)  # Perform MCTS search to get the best action
        
        obs, _ , _ , terminated, info = env_test_post.step(action)
        pot_reward = info['Pot_Reward']
        print(f' Pot_Rew {pot_reward/6}, Load_Id {info["Part_Full"]}, Meth_State {info["Meth_State"]}, Rew {info["reward [ct]"]}, Action {action}')
        print(f"Scenario {EnvConfig.scenario}, Iterations {mcts.iterations}, exploration_weight {mcts.exploration_weight}, total_steps {mcts.total_steps}, maximum_depth {mcts.maximum_depth}")

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
    
    # # ----------------------------------------------Postprocessing------------------------------------------------
    # print("Postprocessing...")
    plot_results(stats_dict_test, EnvConfig, str_id)

if __name__ == '__main__':
    main()



