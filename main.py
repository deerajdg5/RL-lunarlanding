# === PPO Reinforcement Learning Agent for LunarLander-v2 ===

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# === Config ===
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--update_timestep", type=int, default=4000)
    parser.add_argument("--K_epochs", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--logdir", type=str, default="runs/ppo_lunar")
    return parser.parse_args()

# === PPO Policy ===
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, states, actions):
        probs = self.actor(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze()
        return log_probs, values, entropy

# === Memory Buffer ===
class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def clear(self):
        self.__init__()

# === PPO Agent ===
class PPO:
    def __init__(self, state_dim, action_dim, args):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.buffer = RolloutBuffer()
        self.args = args

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action, log_prob = self.policy.act(state)
        value = self.policy.critic(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(torch.tensor(action))
        self.buffer.logprobs.append(log_prob)
        self.buffer.values.append(value.item())
        return action

    def update(self):
        rewards = []
        discounted = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted = 0
            discounted = reward + self.args.gamma * discounted
            rewards.insert(0, discounted)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_states = torch.stack(self.buffer.states)
        old_actions = torch.stack(self.buffer.actions)
        old_logprobs = torch.stack(self.buffer.logprobs)
        old_values = torch.tensor(self.buffer.values)

        advantages = rewards - old_values

        for _ in range(self.args.K_epochs):
            log_probs, values, entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(log_probs - old_logprobs.detach())
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages.detach()
            loss = -torch.min(surr1, surr2).mean() + 0.5 * (rewards - values).pow(2).mean() - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer.clear()

# === Train Function ===
def train(args):
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim, args)
    writer = SummaryWriter(args.logdir)
    scores = []
    timestep = 0

    for episode in range(1, args.max_episodes + 1):
        state = env.reset()[0]
        score = 0
        done = False
        while not done:
            if args.render:
                env.render()

            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)
            score += reward
            state = next_state
            timestep += 1

            if timestep % args.update_timestep == 0:
                agent.update()

        scores.append(score)
        writer.add_scalar("Reward", score, episode)
        print(f"Episode {episode}\tScore: {score:.2f}")

    env.close()
    writer.close()
    return scores

# === Plotting Function ===
def plot_scores(scores):
    plt.plot(scores)
    plt.title("PPO on LunarLander-v2")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    args = get_args()
    scores = train(args)
    plot_scores(scores)
