"""
Simple tabular Q-learning agent.

This file contains the actual learning algorithm for the basic part
of the coursework.

The agent learns a Q-table:

Q[state, action] = expected long-term value of taking that action
from that state.
"""

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_shape,
        num_actions,
        alpha=0.3,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.state_shape = state_shape
        self.num_actions = num_actions

        # alpha controls how strongly we update old Q-values.
        self.alpha = alpha

        # gamma controls how much the agent cares about future rewards.
        self.gamma = gamma

        # epsilon controls exploration.
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)

        # Q-table shape:
        # battery x demand x solar x time x actions
        self.q_table = np.zeros(state_shape + (num_actions,))

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.

        With probability epsilon, the agent explores randomly.
        Otherwise, it chooses the action with the highest Q-value.
        """

        # Explore.
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))

        # Exploit.
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning update.

        Q(s,a) = Q(s,a) + alpha * [target - Q(s,a)]

        where:

        target = reward + gamma * max Q(next_state, next_action)

        If the episode is finished, there is no future value, so the
        target is just the reward.
        """
        action = int(action)

        old_q = self.q_table[state + (action,)]

        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * best_next_q

        td_error = target - old_q

        new_q = old_q + self.alpha * td_error
        self.q_table[state + (action,)] = new_q

        return td_error

    def decay_epsilon(self):
        """
        Reduce exploration slowly after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_action(self, state):
        """
        Greedy action used after training.
        """
        return int(np.argmax(self.q_table[state]))