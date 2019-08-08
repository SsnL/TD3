import signal
import os
import atexit
import logging

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.multiprocessing as mp
from pyvirtualdisplay import Display

import utils


# Runs policy for X episodes and returns average reward
def evaluate_policy_once(env, policy, desc, eval_episodes=10, save_ani=None):
    total_reward = 0
    episode_rewards = []
    animations = []

    for ii in range(eval_episodes):
        obs = env.reset()
        logging.info("{}:\tEvaluate episode {}/{}".format(desc, ii, eval_episodes))
        if save_ani is not None:
            animations.append((ii, None, env.render('rgb_array')))
        done = False
        episode_rewards.append(0)
        while not done:
            with torch.no_grad():
                action = policy.select_action(np.asarray(obs))
            obs, reward, done, _ = env.step(action)
            if save_ani is not None:
                animations.append((ii, reward, env.render('rgb_array')))
            episode_rewards[-1] += reward

    avg_reward = sum(episode_rewards) / eval_episodes
    logging.info("{desc}:\tEvaluation over {eval_episodes} episodes:\t{avg_reward}".format(
        eval_episodes=eval_episodes, avg_reward=avg_reward, desc=desc))

    if save_ani is not None:
        fig = plt.figure()
        ax = fig.gca()

        cumulative_reward = 0

        def plot(data):
            nonlocal cumulative_reward

            # setup axis
            ax.cla()
            ax.axis('off')

            ii, reward, rgb_arr = data

            if reward is None:  # beginning
                cumulative_reward = 0
                reward = 0
            cumulative_reward += reward

            im = ax.imshow(rgb_arr, aspect='equal', interpolation='nearest')
            txt = ax.text(0, 0, '{}\nEpisode: {}/{}\nCurrent R: {:.4f}\nCumulative R: {:.4f}\nEpisode R: {:.4f}'.format(
                desc, ii, eval_episodes, reward, cumulative_reward, episode_rewards[ii]))
            return im, txt

        ani = matplotlib.animation.FuncAnimation(
            fig, plot, frames=animations, blit=True, repeat=False, interval=8)
        ani.save(save_ani, dpi=120)
        plt.close(fig)
        logging.info("{}:\tAnimations saved to {}".format(desc, save_ani))

    return avg_reward


def eval_policy_loop(state, queue):
    utils.logging.configure(state.logging_file, level_prefix='EVAL-')

    virtual_display = Display(visible=0, size=(480, 320))
    virtual_display.start()

    env, policy = utils.make_env_policy(state, train=False)
    evaluations = []

    while True:
        data = queue.get()
        if data is None:
            break

        ckpt_filename, ckpt_dir, desc, eval_episodes, save_ani = data
        policy.load(ckpt_filename, ckpt_dir)
        evaluations.append(evaluate_policy_once(env, policy, desc, eval_episodes, save_ani))
        np.save(os.path.join(state.output_dir, 'results'), evaluations)
    env.close()


def start_eval_worker(state,):
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
    def save_and_eval(policy, num_timesteps):
        desc = 'ts{:07d}'.format(num_timesteps)
        ckpt_folder = os.path.join(state.output_dir, desc)
        utils.mkdir(ckpt_folder)
        policy.save('', ckpt_folder)
        eval_queue.put(('', ckpt_folder, desc, state.eval.episodes, os.path.join(ckpt_folder, 'animation.mp4')))

    def exit_eval_worker():
        logging.info('Exiting eval worker...')
        eval_queue.put(None)
        eval_worker.join()

    atexit.register(exit_eval_worker)

    return save_and_eval  # return handle for save and eval
