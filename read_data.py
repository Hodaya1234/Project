"""
Can read several types of files here:
1. A mat file of the raw data
2. An npz file containing numpy arrays
    a. The raw data in numpy arrays (filename - 'raw_data.npz', containing 'horiz', 'vert')
    b. The segmented data (filename - 'segment_data.npz', containing 'segments', 'segmented_vert', 'segmented_horiz')
    c. The data set for the training (filename - 'data_set.npz')
"""

from scipy import io as sio
import numpy as np


def read_from_file(filename):
    if filename[-3:] == 'mat':
        file = sio.loadmat(filename)
    else:
        file = np.load(filename)
    return file
    # return the dictionary of the arrays


def save_to(data_dictionary, filename, flag):
    if flag == 'mat':
        np.savez(filename, clean_horiz=data_dictionary['clean_horiz'], clean_vert=data_dictionary['clean_vert'],
                 clean_blank=data_dictionary['clean_blank'])
