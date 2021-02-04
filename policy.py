import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path

class Policy(nn.Module):

    def __init__(self, inputs, actor_output, critic_output):
        super(Policy, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(inputs, 32, 3),  # [32, 94, 94]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [32, 47, 47]
            nn.Conv2d(32, 64, 4),  # [64, 44, 44]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [64, 22, 22]
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LogSoftmax(dim=-1)
        )

        # actor's layer
        self.actor_head = nn.Linear(128, actor_output)

        # critic's layer
        self.critic_head = nn.Linear(128, critic_output)


        self.saved_log_probs = []
        self.rewards = []

    
    def forward(self, x):
        
        x= self.pipeline(x)
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.actor_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.critic_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values

    def load_checkpoint(self, params_path):
        if path.exists(params_path):
            self.load_state_dict(torch.load(params_path))
            print("Model params are loaded now")
        else:
            print("Params not found: training from scratch")

    def save_checkpoint(self, params_path):
        torch.save(self.state_dict(), params_path)
        print("Relax, params are saved now")