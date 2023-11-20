"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
"""
import torch
from torchvision.io import read_image, ImageReadMode
from pytorch_msssim import ms_ssim


class Memory():
    def __init__(self, buffer_size=10, device='cpu'):
        super(Memory, self).__init__()
        self.buffer_frame = torch.zeros(
            (buffer_size, 3, 128, 128, 128), device=device)
        self.buffer_audio = torch.zeros((buffer_size, 24576), device=device)
        self.buffer_size = buffer_size

    def write(self, frame, audio):
        self.buffer_frame = torch.cat((frame, self.buffer_frame[:-1]), dim=0)
        self.buffer_audio = torch.cat((audio, self.buffer_audio[:-1]), dim=0)

    def read(self, idx=0):
        return (self.buffer_frame[idx:idx+1], self.buffer_audio[idx:idx+1])

    def __call__(self):
        return (self.buffer_frame, self.buffer_audio)


def reshape_spatial_signal(raw_signal):
    assert raw_signal.shape == (8, 3, 512, 512), "Video signal shape mismatch"
    raw_signal = torch.permute(raw_signal, (1, 0, 2, 3))
    formatted_signal = raw_signal.view(1, 3, 128, 128, 128)
    return formatted_signal


def reshape_audio_signal(raw_signal):
    assert raw_signal.shape == (8, 3072), "Audio signal shape mismatch"
    formatted_signal = raw_signal.view(1, 24576)
    return formatted_signal


def load_referance_frames(
        path='R:/git projects/Agent-76/assets/reward_refrance_images/'):
    ref1, ref2 = read_image(
        path+'r1.png', mode=ImageReadMode.RGB), read_image(
            path+'r2.png', mode=ImageReadMode.RGB)
    return torch.flip(ref1, (0,)).to(dtype=torch.float), torch.flip(
        ref2, (0,)).to(dtype=torch.float)


def load_end_referance_frames(
        path='R:/git projects/Agent-76/assets/reward_refrance_images/'):
    ref1, ref2 = read_image(
        path+'win.png', mode=ImageReadMode.RGB), read_image(
            path+'loss.png', mode=ImageReadMode.RGB)
    return torch.flip(ref1, (0,)).to(dtype=torch.float), torch.flip(
        ref2, (0,)).to(dtype=torch.float)


def get_rois(frame):
    roi_good = frame[:, 226:286, 236:276]
    roi_bad = frame[:, 426:480, 12:43]
    return (roi_good, roi_bad)


def get_end_rois(frame):
    roi_good = frame[:, 226:286, 236:276]
    roi_bad = frame[:, 426:480, 12:43]
    return (roi_good, roi_bad)


def resize_mssim(x, y):
    x = torch.nn.functional.interpolate(x, size=162, mode='nearest')
    y = torch.nn.functional.interpolate(y, size=162, mode='nearest')
    score = ms_ssim(x, y)
    return score


class Reward():
    def __init__(self, device='cpu', boost=1, bias=[0.5, 0.5]):
        super(Reward, self).__init__()
        r1, r2 = load_referance_frames()
        self.pos = get_rois(r1)[0][None, ...].to(device)
        self.neg = get_rois(r2)[1][None, ...].to(device)
        self.metric = resize_mssim
        self.boost = boost
        self.bias = bias

    def __call__(self, state):
        pos = sum([self.metric(self.pos, get_rois(frame)[0][None, ...])
                   for frame in state]) / len(state) * self.bias[0]
        neg = sum([self.metric(self.neg, get_rois(frame)[1][None, ...])
                   for frame in state]) / len(state) * self.bias[1]
        reward = self.boost * 2.5 * torch.round(pos - neg, decimals=1)
        return reward


class Terminate():
    def __init__(self, device='cpu', tau=0.98):
        super(Terminate, self).__init__()
        r1, r2 = load_end_referance_frames()
        self.pos = get_end_rois(r1)[0][None, ...].to(device)
        self.neg = get_end_rois(r2)[1][None, ...].to(device)
        self.metric = resize_mssim
        self.tau = tau

    def __call__(self, state):
        pos = max([self.metric(self.pos, get_end_rois(frame)[0][None, ...])
                   for frame in state]) / len(state) * self.bias[0]
        neg = max([self.metric(self.neg, get_end_rois(frame)[1][None, ...])
                   for frame in state]) / len(state) * self.bias[1]
        return True if pos > self.tau or neg > self.tau else False


if __name__ == '__main__':
    import cv2
    rwd = Reward(boost=10, bias=[0.2, 0.8])
    a, b = load_referance_frames()
    c = get_rois(a)[0]
    print(a.shape, b.shape, c.shape)
    print(rwd.get_reward([a, a, a, a, a, a, a, a]), rwd.get_reward(
        [a, a, a, a, a, a, b, b]), rwd.get_reward([b, b, b, b, b, b, b, b]))
    c = torch.permute(c, (1, 2, 0)).numpy().astype(dtype='uint8')
    print(c.shape)

    while (True):
        cv2.imshow('Bot View', c)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
