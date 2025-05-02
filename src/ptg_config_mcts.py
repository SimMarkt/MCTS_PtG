import math
import random
from typing import Any, List

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[Node] = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_actions())

    def best_child(self, c_param=1.41):
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits) # UCB1 formula
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def most_visited_child(self):
        return max(self.children, key=lambda child: child.visits)

class MCTS:
    def __init__(self, exploration_weight=1.41, iterations=1000):
        self.exploration_weight = exploration_weight
        self.iterations = iterations

    def search(self, initial_state):
        root = Node(initial_state)

        for _ in range(self.iterations):
            node = self._select(root)
            if not node.state.is_terminal():
                node = self._expand(node)
            result = self._simulate(node.state)
            self._backpropagate(node, result)

        return root.most_visited_child().action

    def _select(self, node):
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node):
        tried_actions = [child.action for child in node.children]
        for action in node.state.get_legal_actions():
            if action not in tried_actions:
                new_state = node.state.make_action(action)
                child_node = Node(new_state, parent=node, action=action)
                node.children.append(child_node)
                return child_node
        return node

    def _simulate(self, state):
        current_state = state
        while not current_state.is_terminal():
            actions = current_state.get_legal_actions()
            action = random.choice(actions)
            current_state = current_state.make_action(action)
        return current_state.get_result()

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            if result == node.state.get_current_player():
                node.wins += 1
            elif result == 0.5:  # draw
                node.wins += 0.5
            node = node.parent
