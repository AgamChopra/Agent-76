"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/

System Specific Note: Device 21: Stereo Mix (Realtek(R) Audio) 2,\
                        Channels: 2 input, 0 output
"""
import pyaudio
import torch
import numpy as np
import wave


def list_audio_devices():
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        print(f"Device {i}: {device_info['name']} {device_info['hostApi']}, Channels: \
              {device_info['maxInputChannels']} input, \
                  {device_info['maxOutputChannels']} output")


def get_audio_chunk(chunk_size=8, audio_device=21, device='cuda',
                    channels=2, window_framerate=244, rate=48000,
                    buffer_chunk=1024):
    seconds_per_frame = 1 / window_framerate
    time_to_record = chunk_size * seconds_per_frame

    audio = pyaudio.PyAudio()
    stream = audio.open(rate=rate, channels=channels, format=pyaudio.paInt24,
                        input=True, input_device_index=audio_device,
                        frames_per_buffer=buffer_chunk)
    frames = []

    for _ in range(0, int(rate / buffer_chunk * time_to_record)):
        data = stream.read(buffer_chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.Wave_write('D:/sample.wav')
    wf.setnchannels(2)
    wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt24))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    #audio_data = np.frombuffer(b''.join(frames), dtype=np.int24)

    # return torch.from_numpy(audio_data).to(device=device, dtype=torch.float32)


def test():
    list_audio_devices()

    chunk = 1024
    get_audio_chunk(chunk)  # .detach().cpu()


if __name__ == '__main__':
    test()
