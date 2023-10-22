"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
"""
import cv2
import numpy as np
import win32gui
import win32ui
import win32con
from ctypes import windll
import torch


def window_capture(windowname=None):
    windll.user32.SetProcessDPIAware()
    hwnd = win32gui.FindWindow(None, windowname)

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bottom - top

    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)
    img = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype='uint8')
    img.shape = (h, w, 4)
    img = np.ascontiguousarray(img[:, :, :-1], dtype='float')

    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return img


def get_frame_chunk(chunk_size=8, rescale_size=512, device='cuda',
                    window='Overwatch'):
    frame_chunk = torch.zeros(
        (chunk_size, 3, rescale_size, rescale_size)).to(device)

    for i in range(chunk_size):
        frame = torch.from_numpy(window_capture(window)).to(device)
        frame = torch.permute(frame, (2, 0, 1))[None, ...]
        frame = torch.nn.functional.interpolate(
            frame, (rescale_size, rescale_size), mode='bilinear')
        frame_chunk[i:i+1] = frame

    return frame_chunk


def test():
    while (True):
        chunk = 8
        frame_chunk = get_frame_chunk(chunk_size=chunk).detach().cpu()
        frame_chunk = torch.permute(frame_chunk, (0, 2, 3, 1)).mean(dim=0)

        cv2.imshow('Bot View', frame_chunk.numpy().astype(dtype='uint8'))

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    test()
