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
from src.ptg_utils import load_data, initial_print, config_print, Preprocessing, Postprocessing
from src.rl_config_mcts import AgentConfiguration
from src.ptg_config_env import EnvConfiguration
from src.rl_config_train import TrainConfiguration

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
    MCTSConfig = AgentConfiguration()
    EnvConfig = EnvConfiguration()


    iterations=500
    EnvConfig.scenario = 1
    exploration_weight = 1.41

    TrainConfig = TrainConfiguration()
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

    mcts = MCTS(exploration_weight=exploration_weight, iterations=iterations)

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



