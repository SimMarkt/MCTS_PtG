# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Power-to-Gas Dispatch Optimization using Monte Carlo Tree Search (MCTS)
# GitHub Repository: https://github.com/SimMarkt/MCTS_PtG
#
# ptg_config_mcts: 
# > Contains the source code for the MCTS algorithm
# > Converts the data from 'config_mcts.yaml' into a class object for further processing and usage
# ----------------------------------------------------------------------------------------------------------------

import math
import random
import copy
import yaml

class MCTSNode:
    def __init__(self, env, parent=None, action=None, done=False, remaining_steps=42, total_steps=42, depth=0, maximum_depth=42):
        self.env = env  # deepcopy of the env
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.done = done  # whether this state is terminal
        if remaining_steps < total_steps:
            self.remaining_steps = remaining_steps  # Number of steps remaining for the simulation
        else:
            self.remaining_steps = total_steps
        self.depth = depth
        self.maximum_depth = maximum_depth

    def is_terminal(self):
        return self.done or self.depth >= self.maximum_depth  # Terminal if done or max depth reached

    def is_fully_expanded(self):
        legal_actions = self.get_legal_actions()
        tried_actions = [child.action for child in self.children]
        return set(tried_actions) == set(legal_actions)
    
    def get_legal_actions(self):
        """
        Dynamically determine the legal actions based on the Meth_State.
        :return: A list of legal actions.
        """
        meth_state = self.env.Meth_State  # Access the Meth_State from the environment
        if meth_state == 0:     # 'standby'
            return [0, 1, 2]    # Allows only standby, cooldown, and startup actions
        elif meth_state == 1:   # 'cooldown'
            return [0, 1, 2]    # Allows only standby, cooldown, and startup actions
        elif meth_state == 2:   # 'startup'
            return [0, 1, 3, 4] # Allows only standby, cooldown, and load level after startup (partial load, full load)
        elif meth_state == 3:   # 'partial load'
            return [0, 1, 3, 4] # Allows only standby, cooldown, and load level (partial load, full load)
        elif meth_state == 4:   # 'full load'
            return [0, 1, 3, 4] # Allows only standby, cooldown, and load level (partial load, full load)
        else:
            return list(range(self.env.action_space.n))  # Default to all actions
        
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
    def __init__(self):
        # Load the algorithm configuration from the YAML file
        with open("config/config_mcts.yaml", "r") as env_file:
            mcts_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(mcts_config)


    def search(self, root_env, rootnodeid):
        root_env_copy = copy.deepcopy(root_env)
        root_node = MCTSNode(root_env_copy, total_steps=self.total_steps, maximum_depth=self.maximum_depth)

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
        if node.depth >= self.maximum_depth:  # Prevent expansion beyond max depth
            return node

        legal_actions = node.get_legal_actions()
        tried_actions = [child.action for child in node.children]
        untried_actions = [a for a in legal_actions if a not in tried_actions]

        action = random.choice(untried_actions)
        new_env = copy.deepcopy(node.env)
        obs, reward, terminated, truncated, info = new_env.step(action)
        done = terminated or truncated
        remaining_steps = node.remaining_steps - 1  # Decrease remaining steps by 1 each time we expand (To keep the simulation time equal)
        child_node = MCTSNode(new_env, parent=node, action=action, done=done, remaining_steps=remaining_steps, total_steps=self.total_steps, maximum_depth=self.maximum_depth)
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