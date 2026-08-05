"""
rl_agent.py
Reinforcement Learning Module
Uses Q-Learning to learn the best diagnostic/treatment actions.
"""

import random


class RLAgent:
    def __init__(self,
                 learning_rate=0.1,
                 discount_factor=0.9,
                 exploration_rate=1.0,
                 exploration_decay=0.995,
                 min_exploration=0.01):

        # Q-table
        self.q_table = {}

        # Hyperparameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration

    def get_state_key(self, state):
        """
        Converts a state into a dictionary key.
        Example:
        ["fever","cough"] -> ('cough','fever')
        """
        return tuple(sorted(state))

    def initialize_state(self, state, actions):
        key = self.get_state_key(state)

        if key not in self.q_table:
            self.q_table[key] = {}

            for action in actions:
                self.q_table[key][action] = 0.0

    def choose_action(self, state, actions):
        """
        Epsilon-Greedy Policy
        """

        self.initialize_state(state, actions)

        key = self.get_state_key(state)

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)

        return max(self.q_table[key], key=self.q_table[key].get)

    def update_q_table(self,
                       state,
                       action,
                       reward,
                       next_state,
                       actions):

        self.initialize_state(state, actions)
        self.initialize_state(next_state, actions)

        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        current_q = self.q_table[state_key][action]

        max_future_q = max(self.q_table[next_key].values())

        new_q = current_q + self.alpha * (
            reward +
            self.gamma * max_future_q -
            current_q
        )

        self.q_table[state_key][action] = new_q

    def decay_exploration(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def recommend_action(self, state, actions):
        """
        Returns the best learned action.
        """

        self.initialize_state(state, actions)

        key = self.get_state_key(state)

        return max(self.q_table[key], key=self.q_table[key].get)

    def print_q_table(self):
        print("\n===== Q TABLE =====")

        for state, values in self.q_table.items():
            print(f"\nState: {state}")

            for action, score in values.items():
                print(f"   {action:<25} {score:.2f}")