import numpy as np


def down_sample_frame(data, frame_numbers):
    data_shape = data.shape
    if len(frame_numbers) % 2 == 1:
        frame_numbers.append(frame_numbers[-1] + 1)
    new_frame_numbers = [num for idx,num in enumerate(frame_numbers) if idx % 2 == 0]
    data = data[:, frame_numbers, :]
    new_data = np.mean(data.reshape(data_shape[0], -1, 2, data_shape[2]), 2)
    return new_data, new_frame_numbers



