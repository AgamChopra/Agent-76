"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
"""

# Critic learns a Value function every iteration, Actor learns
# every 10 iterations from all the past 10 steps(memory size) stored in buffer
import os
import torch

from inputs import get_current_state
from utils import Memory, Reward, Terminate
from utils import reshape_audio_signal, reshape_spatial_signal
from model import Actor, Critic
#import action execution functions...


def train(path, lr=1E-2, beta1=0.5, beta2=0.999):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device detected: %s' % device)

    memory_buffer = Memory(device=device)
    reward_function = Reward(device=device)
    end_condition = Terminate(device=device)

    for _ in range(10):
        state = get_current_state(device=device)
        memory_buffer.write(state[0], state[1])

    policy = Actor().to(device)
    value_function = Critic().to(device)

    try:
        print('loading previous run state...', end=" ")
        policy.load_state_dict(torch.load(os.path.join(
            path, 'actor.pth'), map_location=device))
        value_function.load_state_dict(torch.load(
            os.path.join(path, 'critic.pth'), map_location=device))
        print('☑success!')

    except Exception:
        print('☒Unable to locate parameters. Using fresh state.')

    policy_optimizer = torch.optim.AdamW(
        policy.parameters(), lr, betas=(beta1, beta2))
    value_optimizer = torch.optim.AdamW(
        value_function.parameters(), lr, betas=(beta1, beta2))

    while():
        # Train Logic, Memory update, reward handeling, save parameter, ...
        if end_condition(memory_buffer.read()[0]):
            break

        for _ in range(10):
            print('something')
            #get actions from policy
            #execute action command
            #get new state, break from main loop if end condition
            #update buffer
            #calculate previous reward, calculate new state's reward
            #using previous state, action, and reward, predict new state's expected reward using value function
            #calculate error between expected reward and actual reward
            #propogate error, update value function

        #get actions from policy
        #execute action command
        #get new state, break from main loop if end_condition
        #update buffer
        #get expected reward using value function
        #propogate error(some function of -expected_reward), update policy

        #save parameters every nth cycle


if __name__ == '__main__':
    path = ''
    lr = 1E-2
    train(path, lr)