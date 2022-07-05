import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, not_done, unsquashed_action, cost):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, not_done, cost)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, not_done, cost = map(np.stack, zip(*batch))
        return state, action, reward, next_state, not_done, cost

    def get_lasts_transitions(self, batch_size):
        assert len(self) > batch_size, f"You cannot query the last n={batch_size} transitions when len(self)={len(self)}"
        if self.position - batch_size > 0:
            batch = self.buffer[self.position - batch_size - 1: self.position - 1]
        else:
            begin_limit = batch_size - self.position
            batch = self.buffer[begin_limit:] + self.buffer[:self.position]
        state, action, reward, next_state, not_done, cost = map(np.stack, zip(*batch))
        return state, action, reward, next_state, not_done, cost

    def __len__(self):
        return len(self.buffer)
