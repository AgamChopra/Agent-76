"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
"""
import os
import torch
from tqdm import trange
from time import time
import keyboard

from inputs import get_current_state
from utils import Memory, Reward
from utils import reshape_audio_signal, reshape_spatial_signal
from model import Actor, Critic
from relay_actions import Relay


def train(path, lr=1E-2, beta1=0.5, beta2=0.999, discount=0.05):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device detected: %s' % device)

    relay = Relay()
    relay.lookup_ports()
    relay.activate_port(input('Port #: '))

    memory_buffer = Memory(device=device)
    reward_function = Reward(device=device)
    # end_condition = Terminate(device=device)

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

    policy_error = torch.nn.MSELoss()

    start_time = time()

    for _ in range(10):
        state = get_current_state(device=device)
        memory_buffer.write(state[0], state[1])

    while (True):
        for _ in trange(10):
            value_optimizer.zero_grad()
            state = memory_buffer()
            action = policy(reshape_spatial_signal(state[0]),
                            reshape_audio_signal(state[1]))

            # execute action command
            relay(action)

            # get state, update buffer, break from main loop if end condition
            state = get_current_state(device=device)
            memory_buffer.write(state[0], state[1])

            # calculate previous reward, calculate new state's reward
            prev_reward = reward_function(memory_buffer.read(8)[0])
            real_reward = reward_function(memory_buffer.read(9)[0])

            # using previous state, action, and reward, predict new state's
            # expected reward using value function
            expected_reward = value_function(reshape_spatial_signal(
                memory_buffer.read(8)[0]),
                reshape_audio_signal(
                memory_buffer.read(8)[1]),
                action.view(1, 17),
                prev_reward.view(1, 1))

            # calculate error between expected reward and actual reward
            error = policy_error(torch.flatten(real_reward),
                                 torch.flatten(expected_reward))

            # propogate error, update value function
            error.backward(retain_graph=True)
            value_optimizer.step()
            print(real_reward, expected_reward)

        policy_optimizer.zero_grad()
        state = memory_buffer()
        action = policy(reshape_spatial_signal(state[0]),
                        reshape_audio_signal(state[1]))

        # execute action command
        relay(action)
        # get new state, update buffer, break from main loop if end_condition
        state = get_current_state(device=device)
        memory_buffer.write(state[0], state[1])

        # get expected reward using value function
        expected_reward = value_function(reshape_spatial_signal(
            memory_buffer.read(8)[0]),
            reshape_audio_signal(
            memory_buffer.read(8)[1]),
            action.view(1, 17),
            prev_reward.view(1, 1))

        # propogate error(some function of -expected_reward), update policy
        real_reward = reward_function(memory_buffer.read(9)[0])
        error = - (torch.flatten(expected_reward).mean() +
                   discount * torch.flatten(real_reward).mean())
        error.backward()
        policy_optimizer.step()
        print(real_reward, expected_reward)

        if keyboard.is_pressed('p'):
            break

        # !!! save parameters every nth cycle

        print('Elapsed time:', time() - start_time)

    relay.free_port()


if __name__ == '__main__':
    path = ''
    lr = 1E-2
    train(path, lr)
