"""""""""
Codes are heavily borrowed from https://github.com/JamesChuanggg/pytorch-REINFORCE
"""""""""

import argparse, os

import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
from torch.autograd import Variable

import importance_sampling
import REINFORCE

plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='CartPole-v0')
parser.add_argument('--max-steps', type=int, default=200, metavar='N')
parser.add_argument('--num-episodes', type=int, default=1000, metavar='N')
parser.add_argument('--num-trajs', type=int, default=10, metavar='N')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G')
parser.add_argument('--lr', type=float, default=1e-3, metavar='G')
parser.add_argument('--hidden_layer', type=int, default=128, metavar='N')
parser.add_argument('--seed', type=int, default=777, metavar='N',)
parser.add_argument('--reinforce', action ='store_true', help='Use REINFORCE instead of importance sampling')

args = parser.parse_args()

def main():


    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.reinforce:
        agent = REINFORCE.Agent(args, env.observation_space.shape[0], env.action_space)        
    else:
        agent = importance_sampling.Agent(args, env.observation_space.shape[0], env.action_space)

    trajs = []
    result = []

    for i_episode in range(args.num_episodes):

        s_t = torch.Tensor([env.reset()])

        states = []
        actions = []
        log_probs = []
        rewards = []

        for t in range(args.max_steps):
            a_t, log_prob = agent.action(s_t)
            s_t1, r_t, done, _ = env.step(a_t.numpy()[0][0])
            states.append(s_t)
            actions.append(a_t)
            log_probs.append(log_prob)
            rewards.append(r_t)
            s_t = torch.Tensor([s_t1])

            if done:
                break

        if len(trajs) >= args.num_trajs:
            trajs.pop(0)
        
        if args.reinforce:
            ##use most recent trajectory only
            trajs = [] 

        trajs.append((states, actions, rewards, log_probs))
        agent.train_(trajs)

        print("Episode: {}, reward: {}".format(i_episode, sum(rewards)))
        result.append(sum(rewards))


    """plot"""
    plt.plot(range(len(result)), result)
    plt.ylabel('reward')
    plt.xlabel('episodes')
    plt.grid(True)
    plt.show()

    env.close()


if __name__ == '__main__':
    main()
