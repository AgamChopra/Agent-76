"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
"""
import concurrent.futures

from frames import get_frame_chunk, cv2, torch
from audio import get_audio_chunk


def get_frame_audio_chunks(chunk_size=8, device='cpu'):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        frame_task = executor.submit(get_frame_chunk, chunk_size, device)
        audio_task = executor.submit(get_audio_chunk, chunk_size, device)

    frame_chunk = frame_task.result()
    audio_chunk = audio_task.result()

    return frame_chunk, audio_chunk


def get_current_state(chunk_size=8, device='cpu'):
    frame_chunk, audio_chunk = get_frame_audio_chunks(chunk_size, device)
    assert frame_chunk.shape[0] == audio_chunk.shape[0], "chunk size mismatch"
    return (frame_chunk, audio_chunk)


def test():
    while (True):
        data_chunk = get_current_state(8, 'cuda')
        frame_chunk = data_chunk[0].detach().cpu()
        audio_chunk = data_chunk[1].detach().cpu()

        print(frame_chunk.shape, audio_chunk.shape)

        frame_chunk = torch.permute(frame_chunk, (0, 2, 3, 1)).mean(dim=0)

        cv2.imshow('Bot View', frame_chunk.numpy().astype(dtype='uint8'))

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    test()
