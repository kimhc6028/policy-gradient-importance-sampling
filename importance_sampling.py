import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable


def cvt_axis(trajs):
    t_states = []
    t_actions = []
    t_rewards = []
    t_log_probs = []

    for traj in trajs:
        t_states.append(traj[0])
        t_actions.append(traj[1])
        t_rewards.append(traj[2])
        t_log_probs.append(traj[3])

    return (t_states, t_actions, t_rewards, t_log_probs)

def reward_to_value(t_rewards, gamma):

    t_Rs = []

    for rewards in t_rewards:
        Rs = []
        R = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            Rs.insert(0, R)
        t_Rs.append(Rs)
        
    return(t_Rs)

class Network(nn.Module):

    def __init__(self, input_layer, hidden_layer, output_layer):

        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, output_layer)
        

    def forward(self, input_):

        x = F.relu(self.fc1(input_))
        x = F.softmax(self.fc2(x))

        return(x)



class Agent():

    def __init__(self, args, observation_space, action_space):

        self.model = Network(observation_space, args.hidden_layer, action_space.n)
        self.gamma = args.gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.model.train()


    def action(self, state):
        
        probs = self.model(Variable(state))
        action = probs.multinomial().data
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()

        return(action, log_prob)


    def cal_log_prob(self, state, action):

        probs = self.model(Variable(state))
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()

        return(log_prob)


    def train_(self, trajs):
        
        t_states, t_actions, t_rewards, t_log_probs = cvt_axis(trajs)
        t_Rs = reward_to_value(t_rewards, self.gamma)

        Z = 0
        b = 0
        losses = []
        Z_s = []

        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            p_log_prob = 0
            q_log_prob = 0
            for t in range(len(Rs)):
                p_log_prob += (self.cal_log_prob(states[t], actions[t])).data.numpy()
                q_log_prob += log_probs[t].data.numpy()
            Z_ = math.exp(p_log_prob) / math.exp(q_log_prob)
            Z += Z_
            Z_s.append(Z_)
            b += Z_ * sum(Rs) / len(Rs)
        b = b / Z


        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            loss = 0.

            for t in range(len(Rs)):
                loss = loss - (log_probs[t] * (Variable(Rs[t] - b).expand_as(log_probs[t]))).sum()

            Z_ = Z_s.pop(0)
            loss = loss / Z_
            losses.append(loss)
            
        loss = sum(losses) / Z

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()


        
