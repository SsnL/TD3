import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.multiprocessing as mp
import gym
from gym import wrappers
import argparse
import signal
import os

import utils
import TD3
import OurDDPG
import DDPG


from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(480, 320))
virtual_display.start()

# Runs policy for X episodes and returns average reward
def evaluate_policy_once(policy, eval_episodes=10, save_ani=None):
    avg_reward = 0.

    def eval_helper():
        nonlocal avg_reward

        for ii in range(eval_episodes):
            obs = env.reset()
            yield ii, env.render('rgb_array')
            done = False
            while not done:
                with torch.no_grad():
                    action = policy.select_action(np.asarray(obs))
                obs, reward, done, _ = env.step(action)
                yield ii, env.render('rgb_array')
                avg_reward += reward

        avg_reward /= eval_episodes
        return

    if save_ani is None:
        for _ in eval_helper():
            pass
    else:
        fig = plt.figure()
        ax = fig.gca()

        def plot(data):
            # setup axis
            ax.cla()
            ax.axis('off')

            ii, rgb_arr = data

            # imshow, no need to flip
            im = ax.imshow(rgb_arr, aspect='equal', interpolation='nearest')
            txt = ax.text(0, 0, 'Episode: {}/{}'.format(ii, eval_episodes))
            # ax.set_title(self.get_title(timestep))
            return im, txt

        ani = matplotlib.animation.FuncAnimation(
            fig, plot, frames=list(eval_helper()), blit=True, repeat=False, interval=10)
        ani.save(save_ani, dpi=300)

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


def eval_policy_loop(args, file_name, queue):
    env, policy = make_env_policy(args.env_name, args.policy_name, args.lr, args.eval_seed)
    evaluations = []

    while True:
        data = queue.get()

        if data is None:
            return

        ckpt_filename, ckpt_dir, eval_episodes, save_ani = data
        policy.load(ckpt_filename, ckpt_dir)
        evaluations.append(evaluate_policy_once(policy, eval_episodes, save_ani))
        np.save("./results/%s" % (file_name), evaluations)


def make_env_policy(env_name, policy_name, lr, seed=None):
    env = gym.make(env_name)

    # Set seeds
    if seed is not None:
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action, lr=lr)
    elif policy_name == "OurDDPG":
        policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)

    return env, policy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")                             # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v1")                     # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_seed", default=21320, type=int)                     # Sets Gym, PyTorch and Numpy eval seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)                 # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=8e3, type=int)                       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)                   # Max time steps to run environment for
    # parser.add_argument("--max_timesteps_per_episode", default=2000, type=int)    # Max time steps per episode to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)                    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)                      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                     # Discount factor
    parser.add_argument("--lr", default=0.001, type=float)                          # Learning rate
    parser.add_argument("--tau", default=0.005, type=float)                         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)                  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                    # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)                       # Frequency of delayed policy updates
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    env, policy = make_env_policy(args.env_name, args.policy_name, args.lr, args.seed)
    print('Environment:', env)

    replay_buffer = utils.ReplayBuffer()

    # Eval worker
    mp_ctx = mp.get_context('spawn')
    eval_queue = mp_ctx.Queue()
    eval_worker = utils.multiprocessing.ErrorTrackingProcess(
        ctx=mp_ctx,
        target=eval_policy_loop,
        args=(args, file_name, eval_queue))
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
        ckpt_folder = 'ts{:012d}'.format(num_timesteps)
        utils.mkdir(ckpt_dir)
        policy.save('', ckpt_folder)
        eval_queue.put((num_timesteps, ckpt_folder))

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    while total_timesteps < args.max_timesteps:

        if done:
            if total_timesteps > 0:
                print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (total_timesteps, episode_num, episode_timesteps, episode_reward))
                if args.policy_name == "TD3":
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
                else:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                save_and_eval(total_timesteps)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    save_and_eval(total_timesteps)
