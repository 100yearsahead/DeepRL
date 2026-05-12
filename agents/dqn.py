"""
Simple DQN agent for the continuous microgrid environment.

DQN idea:
- The state is continuous, so a Q-table is not practical.
- A neural network estimates Q-values for each action.
- The action with the highest Q-value is selected.

This implementation includes:
1. Replay buffer
2. Target network
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Small neural network that maps:

    state -> Q-values for each action

    Output:
    [Q(charge), Q(idle), Q(discharge)]
    """

    def __init__(self, state_size, num_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Stores past experiences.

    Each experience is:
    (state, action, reward, next_state, done)

    DQN trains from random batches of these experiences.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size,
        num_actions,
        gamma=0.95,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        replay_size=10000,
        seed=42,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_size = state_size
        self.num_actions = num_actions

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network: this is the network we train every step.
        self.q_net = QNetwork(state_size, num_actions).to(self.device)

        # Target network: this is a slower copy used to calculate targets.
        self.target_net = QNetwork(state_size, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(capacity=replay_size)

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state_t)

        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        """
        Store one transition in replay memory.
        """
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Train the Q-network from one random batch of replay memory.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q-value for the action actually taken.
        current_q = self.q_net(states).gather(1, actions)

        # Target Q-value:
        # reward + discounted best next action value.
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        """
        Copy online network weights into the target network.
        """
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        """
        Reduce exploration after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_action(self, state):
        """
        Use the learned policy without exploration.
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state_t)

        return int(torch.argmax(q_values).item())