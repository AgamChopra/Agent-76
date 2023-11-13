"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
"""
import torch
from torchvision.io import read_image, ImageReadMode


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


def load_referance_frames(path='R:/git projects/Agent-76/assets/reward_refrance_images/'):
    ref1, ref2 = read_image(
        path+'r1.png', mode=ImageReadMode.RGB), read_image(
            path+'r2.png', mode=ImageReadMode.RGB)
    return torch.flip(ref1, (0,)).to(dtype=torch.float), torch.flip(
        ref2, (0,)).to(dtype=torch.float)


def get_rois(frame):
    roi_good = frame[:, 226:286, 236:276]
    roi_bad = frame[:, 426:480, 12:43]
    return roi_good.mean(dim=0), roi_bad.mean(dim=0)


######
class Reward():
    def __init__(self):
        super(Reward, self).__init__()

    def get_reward(state):
        return


if __name__ == '__main__':
    import cv2

    a, b = load_referance_frames()
    print(a.shape, b.shape)
    a, _ = get_rois(a)

    while(True):
        cv2.imshow('Bot View', a.numpy().astype(dtype='uint8'))
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
