import torch
import random
import gym as openai_gym
import numpy as np

import algorithms

from .replay_buffer import ReplayBuffer  # noqa: F401


# Wrap the ssnl_utils module

import os
import sys
import importlib

from .ssnl_utils import *  # noqa: F401


class CustomFinder(importlib.abc.MetaPathFinder):
    __this_module_dir = os.path.split(__file__)[0]
    __wrappes_module_dir = os.path.join(__this_module_dir, 'ssnl_utils')

    @classmethod
    def find_spec(cls, fullname, path, target=None):
        # NB: It is possible that this only works properly if one does `import utils.gym`
        #     before `from .ssnl_utils import gym` in this script (i.e., not
        #     running things twice due to proper caching). However, since this
        #     wrapping of `ssnl_utils` should be completely invisible to users
        #     as well as in this script. So, to import `ssnl_utils.gym`, we
        #     expect the corresponding line of code be `from . import gym`. Thus
        #     this problem is avoided.
        if path is not None and len(path) == 1 and path[0].startswith(cls.__this_module_dir):
            if not path[0].startswith(cls.__wrappes_module_dir):
                path = [os.path.join(cls.__wrappes_module_dir, path[0][len(cls.__this_module_dir):])]
            if fullname.startswith('utils.ssnl_utils.'):
                fullname = 'utils.' + fullname[len('utils.ssnl_utils.'):]  # make cache happy
            return importlib.machinery.PathFinder.find_spec(fullname, path, target)
        return None


# hack so import utils.gym work (.gym is not automatically imported in ssnl_utils)
sys.meta_path.insert(0, CustomFinder())


def make_env_policy(state, train, env_terrain_data=None):
    env = openai_gym.make(
        state.env.name,
        agent_kwargs=dict(LEG_DOWN_COEF=state.agent.leg_down,
                          LEG_W_COEF=state.agent.leg_width,
                          LEG_H_COEF=state.agent.leg_height),
        difficulty=state.env.difficulty,
        fix_terrain=state.env.fix_terrain,
        terrain_seed=state.env.terrain_seed,
        terrain_data=env_terrain_data)

    max_episode_steps = getattr(env, '_max_episode_steps', None)
    if state.env.max_episode_steps is not None:
        from openai_gym.wrappers.time_limit import TimeLimit
        env = TimeLimit(env.unwrapped, max_episode_steps=state.env.max_episode_steps)
        max_episode_steps = state.env.max_episode_steps

    env.max_episode_steps = max_episode_steps

    # Set seeds
    if train:
        seed = state.seed
        device = state.device
        if state.reward_scale != 1:
            reward_scale = state.reward_scale
            from . import gym
            env = gym.wrappers.TransformReward(env, lambda r: reward_scale * r)
    else:
        seed = state.eval.seed
        device = state.eval.device

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env.reset()
    env.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = algorithms.TD3(state_dim, action_dim, max_action, lr=state.lr, device=device, init=state.init)
    # if args.policy_name == "TD3":
    #     policy = algorithms.TD3(state_dim, action_dim, max_action, lr=args.lr, device=device)
    # elif args.policy_name == "OurDDPG":
    #     policy = algorithms.DDPG(state_dim, action_dim, max_action)
    # elif args.policy_name == "DDPG":
    #     policy = algorithms.DDPG(state_dim, action_dim, max_action)

    return env, policy
