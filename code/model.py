"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
"""
import torch
from torch import nn
# Modify code to input memory buffer as batch of states, then use mlp to
# reduce dimensionality from 100(buffer size) to 1 for Actor ?and? ?Critic?...


def normalize(x, mode='min-max', epsilon=1E-9):
    if mode == 'min-max':
        return (x - x.min()) / (x.max() - x.min() + epsilon)
    elif mode == 'max':
        return x / (x.max() + epsilon)
    elif mode == 'std':
        return (x - x.mean()) / (torch.std(x) + epsilon)


class Block(nn.Module):
    def __init__(self, ic, hc, oc, norm=nn.InstanceNorm3d):
        super(Block, self).__init__()
        self.layer = nn.Sequential(nn.Conv3d(
            in_channels=ic, out_channels=hc, kernel_size=3),
            nn.ReLU(), norm(hc), nn.Conv3d(
            in_channels=hc, out_channels=oc, kernel_size=2),
            nn.ReLU(), norm(oc))

    def forward(self, x):
        y = self.layer(x)
        return y


class Actor(nn.Module):
    def __init__(self, norm=nn.InstanceNorm3d, buffer_size=10):
        super(Actor, self).__init__()
        self.buffer_size = buffer_size
        self.l1 = nn.Sequential(Block(3, 8, 8, norm), nn.Conv3d(
            in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.ReLU(), norm(8))
        self.l2 = nn.Sequential(Block(8, 16, 16, norm), nn.Conv3d(
            in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(), norm(16))
        self.l3 = nn.Sequential(Block(16, 32, 32, norm), nn.Conv3d(
            in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.ReLU(), norm(32))
        self.l4 = nn.Sequential(Block(33, 64, 64, norm), nn.Conv3d(
            in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(), norm(64))
        self.l5 = nn.Sequential(Block(64, 128, 128, norm), nn.Conv3d(
            in_channels=128, out_channels=128, kernel_size=2),
            nn.ReLU())
        self.out = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=17, kernel_size=1),
            nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Linear(24576, 2197), nn.ReLU(), nn.InstanceNorm1d(2197))
        self.reduce = nn.Sequential(nn.InstanceNorm1d(17 * buffer_size, 64),
                                    nn.Linear(17 * buffer_size, 64), nn.ReLU(),
                                    nn.InstanceNorm1d(64),
                                    nn.Linear(64, 17))

    def forward(self, x_frames, x_audio):
        y = self.l1(x_frames)
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(
            torch.cat((y, self.mlp(x_audio).view(
                self.buffer_size, 1, 13, 13, 13)), dim=1))
        y = self.l5(y)
        y = self.out(y).view(1, 17 * self.buffer_size)
        y = self.reduce(y)
        return y.view((17))


class Critic(nn.Module):
    def __init__(self, norm=nn.InstanceNorm3d):
        super(Critic, self).__init__()
        self.l1 = nn.Sequential(Block(3, 4, 4, norm), nn.Conv3d(
            in_channels=4, out_channels=4, kernel_size=2, stride=2),
            nn.ReLU(), norm(4))
        self.l2 = nn.Sequential(Block(4, 8, 8, norm), nn.Conv3d(
            in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.ReLU(), norm(8))
        self.l3 = nn.Sequential(Block(8, 16, 16, norm), nn.Conv3d(
            in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(), norm(16))
        self.l4 = nn.Sequential(Block(19, 32, 32, norm), nn.Conv3d(
            in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.ReLU(), norm(32))
        self.l5 = nn.Sequential(Block(32, 64, 64, norm), nn.Conv3d(
            in_channels=64, out_channels=64, kernel_size=2),
            nn.ReLU())
        self.out = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1)
        self.mlp_aud = nn.Sequential(
            nn.Linear(24576, 2197), nn.ReLU(), nn.InstanceNorm1d(2197))
        self.mlp_act = nn.Sequential(
            nn.Linear(17, 2197), nn.ReLU(), nn.InstanceNorm1d(2197))
        self.mlp_rwd = nn.Sequential(
            nn.Linear(1, 2197), nn.ReLU(), nn.InstanceNorm1d(2197))

    def forward(self, x_frames, x_audio, x_action, x_reward):
        y_aud = self.mlp_aud(x_audio).view(1, 1, 13, 13, 13)
        y_act = self.mlp_act(x_action).view(1, 1, 13, 13, 13)
        y_rwd = self.mlp_rwd(x_reward).view(1, 1, 13, 13, 13)
        y = self.l1(x_frames)
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(torch.cat((y, y_aud, y_act, y_rwd), dim=1))
        y = self.l5(y)
        y = self.out(y)
        return y.view((1))


def test(device='cpu'):
    frames = torch.rand((10, 3, 128, 128, 128), device=device)
    audio = torch.rand((10, 24576), device=device)
    reward = torch.rand(1, device=device)

    policy = Actor().to(device)
    value_function = Critic().to(device)

    action = policy(frames, audio)
    value = value_function(
        frames[9:10], audio[9:10], action.view(1, 17), reward.view(1, 1))

    print('.....')
    print(frames.shape, audio.shape)
    print(action.shape, reward.shape)
    print(value.shape)


if __name__ == '__main__':
    test('cuda')
