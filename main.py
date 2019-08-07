import random
import signal
import os
import atexit
import logging
import re
import time

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.multiprocessing as mp
import gym

import utils
from utils import state
import algorithms

from bipedal_walker import BipedalWalker

gym.envs.register(
    id='BipedalWalkerCustom-v2',
    entry_point=BipedalWalker,
    max_episode_steps=1600,
    reward_threshold=300,
)

gym.envs.register(
    id='BipedalWalkerHardcoreCustom-v2',
    entry_point=BipedalWalker,
    max_episode_steps=2000,
    reward_threshold=300,
    kwargs=dict(hardcore=True),
)


####################
# Output and logging
####################


def set_start_time():
    state.start_time = time.strftime(r"%Y-%m-%d %H:%M:%S")

state.register_parse_hook(set_start_time)

state.add_option('output_base_dir', type=str,
                 default='./results/', desc='Base directory to store outputs')

state.add_option('output_folder', type=utils.types.make_optional(str),
                 default=None, desc='Folder to store outputs')


def set_output_folder():
    if state.output_folder is None:
        config_base = os.path.splitext(os.path.basename(state.config))[0]
        output_folder_parts = [config_base, state.env.name, 'seed{}'.format(state.seed)]

        with state.overwrite():
            time_suffix = re.sub('[^0-9a-zA-Z]+', '_', state.start_time)
            output_folder_parts.append(time_suffix)
            state.output_folder = '_'.join(output_folder_parts)
    state.output_dir = os.path.join(state.output_base_dir, state.output_folder)

    # if os.path.isdir(state.output_dir) and not state.overwrite_output:
    #     raise RuntimeError("Output directory {} exists, exiting.".format(state.output_dir))
    utils.mkdir(state.output_dir)


state.register_parse_hook(set_output_folder)


state.add_option('logging_file', type=utils.types.make_optional(str),
                 default='output.log', desc='Filename to log outputs')


def config_logging():
    if state.logging_file is not None:
        with state.overwrite():
            state.logging_file = os.path.join(state.output_dir, state.logging_file)
    utils.logging.configure(state.logging_file)
    state.logging_configured = True

state.register_parse_hook(config_logging)


#############
# Environment
#############

with state.option_namespace('env'):
    state.add_option("name", type=str, default='BipedalWalker-v2', desc="OpenAI gym environment name")
    state.add_option("max_episode_steps", default=None, type=utils.types.make_optional(int),
                     desc='Max number of steps per episode')

#######
# Agent
#######

with state.option_namespace('agent'):
    state.add_option("leg_down", type=float, default=8)
    state.add_option("leg_width", type=float, default=8)
    state.add_option("leg_height", type=float, default=34)

##########
# Training
##########

# state.add_option("algorithm", default="TD3", desc="Policy algorithm name")
state.add_option("batch_size", default=100, type=int, desc='Batch size for both actor and critic')
state.add_option("max_timesteps", default=1e6, type=int, desc='Max total number of steps')
state.add_option("start_timesteps", default=1e4, type=int,
                 desc='Number of steps that purely random policy is run for')
state.add_option("expl_noise_std", default=0.1, type=float, desc='Std of Gaussian exploration noise')


state.add_option("discount", default=0.99, type=float, desc='Discount factor')
state.add_option("lr", default=0.001, type=float, desc='Learning rate')
state.add_option("tau", default=0.005, type=float, desc='Target network update rate')

state.add_option("device", type=torch.device, default="cuda:0", desc='Training device')
state.add_option("seed", default=0, type=int, desc='Training seed')

with state.option_namespace('eval'):
    state.add_option("device", type=torch.device, default="cuda:1", desc='Eval device')
    state.add_option("seed", default=21320, type=int, desc='Eval seed')
    state.add_option("freq", default=5e4, type=int, desc='Frequency for evaluation')

state.add_option("policy_noise", default=0.2, type=float, desc='Noise added to target policy during critic update')
state.add_option("noise_clip", default=0.5, type=float, desc='Range to clip target policy noise')
state.add_option("policy_freq", default=2, type=int, desc='Frequency of delayed policy updates')


from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(480, 320))
virtual_display.start()

# Runs policy for X episodes and returns average reward


def evaluate_policy_once(env, policy, desc, eval_episodes=10, save_ani=None):
    avg_reward = 0
    animations = []

    for ii in range(eval_episodes):
        obs = env.reset()
        logging.info("{}:\tEvaluate episode {}/{}".format(desc, ii, eval_episodes))
        if save_ani is not None:
            animations.append((ii, env.render('rgb_array')))
        done = False
        while not done:
            with torch.no_grad():
                action = policy.select_action(np.asarray(obs))
            obs, reward, done, _ = env.step(action)
            if save_ani is not None:
                animations.append((ii, env.render('rgb_array')))
            avg_reward += reward

    avg_reward /= eval_episodes

    logging.info("{desc}:\tEvaluation over {eval_episodes} episodes:\t{avg_reward}".format(
        eval_episodes=eval_episodes, avg_reward=avg_reward, desc=desc))

    if save_ani is not None:
        fig = plt.figure()
        ax = fig.gca()

        def plot(data):
            # setup axis
            ax.cla()
            ax.axis('off')

            ii, rgb_arr = data

            im = ax.imshow(rgb_arr, aspect='equal', interpolation='nearest')
            txt = ax.text(0, 0, 'Episode: {}/{}'.format(ii, eval_episodes))
            return im, txt

        ani = matplotlib.animation.FuncAnimation(
            fig, plot, frames=animations, blit=True, repeat=False, interval=8)
        ani.save(save_ani, dpi=100)
        plt.close(fig)
        logging.info("{}:\tAnimations saved to {}".format(desc, save_ani))

    return avg_reward


def eval_policy_loop(state, queue):
    utils.logging.configure(state.logging_file, level_prefix='EVAL-')
    env, policy = make_env_policy(state, train=False)
    evaluations = []

    while True:
        data = queue.get()

        if data is None:
            return

        ckpt_filename, ckpt_dir, desc, eval_episodes, save_ani = data
        policy.load(ckpt_filename, ckpt_dir)
        evaluations.append(evaluate_policy_once(env, policy, desc, eval_episodes, save_ani))
        np.save(os.path.join(state.output_dir, 'results'), evaluations)


def make_env_policy(state, train):
    env = gym.make(state.env.name, LEG_DOWN_COEF=state.agent.leg_down,
                   LEG_W_COEF=state.agent.leg_width, LEG_H_COEF=state.agent.leg_height)
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

    if seed is not None:
        env.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

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


state.set_desc("TD3")

if __name__ == "__main__":
    state.parse_options()
    logging.info('')
    logging.info(state)
    logging.info('')

    env, policy = make_env_policy(state, train=True)
    logging.info('Environment: {}'.format(env))

    replay_buffer = utils.ReplayBuffer()

    # Eval worker
    mp_ctx = mp.get_context('spawn')
    eval_queue = mp_ctx.Queue()
    eval_worker = utils.multiprocessing.ErrorTrackingProcess(
        ctx=mp_ctx,
        target=eval_policy_loop,
        args=(state, eval_queue))
    eval_worker.daemon = True
    eval_worker.start()

    # raise error when eval_worker errors
    def set_SIGCHLD_handler_for_eval_worker():
        previous_handler = signal.getsignal(signal.SIGCHLD)
        if not callable(previous_handler):
            # This doesn't catch default handler, but SIGCHLD default handler is a
            # no-op.
            previous_handler = None

        def handler(signum, frame):
            if getattr(eval_worker, 'exception_wrapper', None) is not None:
                raise eval_worker.exception_wrapper.reconstruct()
            if previous_handler is not None:
                previous_handler(signum, frame)

        signal.signal(signal.SIGCHLD, handler)

    set_SIGCHLD_handler_for_eval_worker()

    # save and eval call
    def save_and_eval(num_timesteps):
        desc = 'ts{:08d}'.format(num_timesteps)
        ckpt_folder = os.path.join(state.output_dir, desc)
        utils.mkdir(ckpt_folder)
        policy.save('', ckpt_folder)
        eval_queue.put(('', ckpt_folder, desc, 5, os.path.join(ckpt_folder, 'animation.mp4')))

    # Evaluate untrained policy
    save_and_eval(0)

    def exit_eval_worker():
        logging.info('Exiting eval worker...')
        eval_queue.put(None)
        eval_worker.join()

    atexit.register(exit_eval_worker)

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    while total_timesteps < state.max_timesteps:

        if done:
            if total_timesteps > 0:
                logging.info(utils.format_str(total=total_timesteps, episode=episode_num,
                                              episode_T=episode_timesteps, reward=episode_reward))
                # if args.policy_name == "TD3":
                policy.train(
                    replay_buffer, episode_timesteps, state.batch_size,
                    state.discount, state.tau, state.policy_noise,
                    state.noise_clip, state.policy_freq)
                # else:
                #     policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

            # Evaluate episode
            if timesteps_since_eval >= state.eval.freq:
                timesteps_since_eval %= state.eval.freq
                save_and_eval(total_timesteps)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < state.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(obs, expl_noise_std=state.expl_noise_std)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = int(done)  # 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    save_and_eval(total_timesteps)
