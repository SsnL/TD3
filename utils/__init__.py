import torch
import random
import gym
import numpy as np

import algorithms

from .ssnl_utils import *  # noqa: F401


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def __len__(self):
        return len(self.storage)


def make_env_policy(state, train):
    env = gym.make(state.env.name,
                   agent_kwargs=dict(LEG_DOWN_COEF=state.agent.leg_down,
                                     LEG_W_COEF=state.agent.leg_width,
                                     LEG_H_COEF=state.agent.leg_height),
                   hardcore=state.env.hardcore,
                   fix_terrain=state.env.fix_terrain)
    if state.env.max_episode_steps is not None:
        from gym.wrappers.time_limit import TimeLimit
        env = TimeLimit(env.unwrapped, max_episode_steps=state.env.max_episode_steps)

    # Set seeds
    if train:
        seed = state.seed
        device = state.device
    else:
        seed = state.eval.seed
        device = state.eval.device

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if state.env.fix_terrain:
        env_seed = state.seed
    else:
        env_seed = seed
    env.seed(env_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = algorithms.TD3(state_dim, action_dim, max_action, lr=state.lr, device=device)
    # if args.policy_name == "TD3":
    #     policy = algorithms.TD3(state_dim, action_dim, max_action, lr=args.lr, device=device)
    # elif args.policy_name == "OurDDPG":
    #     policy = algorithms.DDPG(state_dim, action_dim, max_action)
    # elif args.policy_name == "DDPG":
    #     policy = algorithms.DDPG(state_dim, action_dim, max_action)

    return env, policy
