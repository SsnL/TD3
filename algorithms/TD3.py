import os

import torch
import torch.nn as nn
import torch.nn.functional as F

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    return net.apply(init_func)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    class Q(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(Critic.Q, self).__init__()

            # Q architecture
            self.l1 = nn.Linear(state_dim + action_dim, 400)
            self.l2 = nn.Linear(400, 300)
            self.l3 = nn.Linear(300, 1)

        def forward(self, xu):
            x = F.relu(self.l1(xu))
            x = F.relu(self.l2(x))
            x = self.l3(x)
            return x

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.q1 = Critic.Q(state_dim, action_dim)
        self.q2 = Critic.Q(state_dim, action_dim)

    def forward(self, x, u, get_q2=True):
        xu = torch.cat([x, u], 1)
        if get_q2:
            return self.q1(xu), self.q2(xu)
        else:
            return self.q1(xu)


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, lr, device=default_device, init="kaiming"):
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, amsgrad=True)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, amsgrad=True)

        if os.path.isdir(init):
            self.load(init)
        else:
            init_weights(self.actor, init)
            init_weights(self.critic, init)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

        self.max_action = max_action

    def select_action(self, state, expl_noise_std=0):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.get_default_dtype(), device=self.device).reshape(1, -1)
            action = self.actor(state)
            if expl_noise_std != 0:
                action.add_(torch.empty_like(action).normal_(0, expl_noise_std))
                action.clamp_(-self.max_action, self.max_action)
            return action.cpu().detach().numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.as_tensor(x, dtype=torch.get_default_dtype(), device=self.device)
            action = torch.as_tensor(u, dtype=torch.get_default_dtype(), device=self.device)
            next_state = torch.as_tensor(y, dtype=torch.get_default_dtype(), device=self.device)
            done = torch.as_tensor(1 - d, dtype=torch.get_default_dtype(), device=self.device)
            reward = torch.as_tensor(r, dtype=torch.get_default_dtype(), device=self.device)

            # Select action according to policy and add clipped noise
            noise = torch.empty(u.shape, device=self.device).normal_(0, policy_noise).clamp_(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp_(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + done * discount * target_Q.detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic(state, self.actor(state), get_q2=False).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                with torch.no_grad():
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data = torch.lerp(target_param, param, tau)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data = torch.lerp(target_param, param, tau)

    def save(self, directory):
        torch.save(
            dict(net=self.actor.state_dict(), optim=self.actor_optimizer.state_dict()),
            os.path.join(directory, 'actor.pth'))
        torch.save(
            dict(net=self.critic.state_dict(), optim=self.critic_optimizer.state_dict()),
            os.path.join(directory, 'critic.pth'))

    def load(self, directory):
        actor_state_dict = torch.load(os.path.join(directory, 'actor.pth'), map_location=self.device)
        if set(actor_state_dict.keys()) == {'net', 'optim'}:
            self.actor_optimizer.load_state_dict(actor_state_dict['optim'])
            actor_state_dict = actor_state_dict['net']
        self.actor.load_state_dict(actor_state_dict)
        self.actor_target.load_state_dict(actor_state_dict)
        critic_state_dict = torch.load(os.path.join(directory, 'critic.pth'), map_location=self.device)
        if set(critic_state_dict.keys()) == {'net', 'optim'}:
            self.critic_optimizer.load_state_dict(critic_state_dict['optim'])
            critic_state_dict = critic_state_dict['net']
        self.critic.load_state_dict(critic_state_dict)
