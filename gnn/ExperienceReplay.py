import numpy as np
from collections import deque

class ExperienceReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

# class RLAgent:
#     def __init__(self, state_dim, action_dim, epsilon, buffer_size):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.epsilon = epsilon
#         self.buffer = ExperienceReplayBuffer(buffer_size)

#     def select_action(self, state, online_network):
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(self.action_dim)
#         else:
#             return np.argmax(online_network.predict(state))

#     def add_experience(self, state, action, next_state, reward, done):
#         self.buffer.add((state, action, next_state, reward, done))