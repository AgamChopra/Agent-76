"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
"""
import torch


class Memory():
    def __init__(self, buffer_size=100):
        super(Memory, self).__init__()
        self.buffer = []
        self.buffer_size = buffer_size

    def write(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def read(self, idx):
        if self.buffer_size > idx and idx >= 0:
            return self.buffer[idx]
        elif idx == -1 or idx >= self.buffer_size:
            return self.buffer[-1]
        else:
            return self.buffer[0]


def reshape_spatial_signal(raw_signal):
    assert raw_signal.shape == (8, 3, 512, 512), "Video signal shape mismatch"
    raw_signal = torch.permute(raw_signal, (1, 0, 2, 3))
    formatted_signal = raw_signal.view(1, 3, 128, 128, 128)
    return formatted_signal


def reshape_audio_signal(raw_signal):
    assert raw_signal.shape == (8, 3072), "Audio signal shape mismatch"
    formatted_signal = raw_signal.view(1, 24576)
    return formatted_signal


######
def get_reward(state):
    return
