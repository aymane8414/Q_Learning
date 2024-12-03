import numpy as np
import pickle
import os
import random
import pandas as pd

ALPHA = 0.10
GAMMA = 0.99
EPSILON = 0.40
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01

STATE = [10, 2, 10]
ACTIONS = [0, 1, 2]

Q_TABLE_FILE = "q_table.pkl"


class QLearningAgent:
    def __init__(self):
        self.q_table = self.load_q_table()

    def discretize_state(self, state):
        bins = [np.linspace(0, 1, num=b) for b in STATE]
        return tuple(np.digitize(s, bins[i]) - 1 for i, s in enumerate(state))

    def choose_action(self, state):
        if random.random() < EPSILON:
            return random.choice(ACTIONS)
        discrete_state = self.discretize_state(state)
        return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state, action, reward, next_state, done):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        q_value = self.q_table[discrete_state][action]
        max_next_q = np.max(self.q_table[discrete_next_state])

        target = reward + (GAMMA * max_next_q * (1 - done))
        self.q_table[discrete_state][action] += ALPHA * (target - q_value)

        self.save_q_table()

    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, "rb") as f:
                return pickle.load(f)
        return np.zeros(tuple(STATE) + (len(ACTIONS),))

    def save_q_table(self):
        with open(Q_TABLE_FILE, "wb") as f:
            pickle.dump(self.q_table, f)
