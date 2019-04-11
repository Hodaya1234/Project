import numpy as np
from matplotlib import pyplot as plt
import segment
import scipy.io as sio
from skimage.segmentation import find_boundaries

def plot_losses(train_losses, validation_losses, test_losses, n_data_sets):
    places = [i for i in range(len(train_losses)) if i % n_data_sets == 0]
    labels = [str(i) for i in range(int(np.ceil(len(train_losses / n_data_sets))))]
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(validation_losses, label="validation")
    plt.plot(test_losses, label="test (original examples left out)")
    plt.xticks(places, labels)
    plt.legend()
    plt.title('Losses as a Function of Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def plot_weights(model_state, seg_mask):
    """
    Take a weight matrix of the size [layer2 X layer1] and plot the image
    :param model_state: dictionary of the weight of the model
    :param seg_mask the mask of the segments - to recreate the original data
    :return: a plot of the "heat map" of the first layer
    """
    n_seg = len(np.unique(seg_mask)) - 1
    w1 = model_state[next(iter(model_state))]
    squeeze_w = np.sum(w1, axis=0)
    squeeze_w = np.reshape(squeeze_w, (n_seg, -1))
    image = segment.recreate_image(seg_mask, squeeze_w)
    return image


def plot_frame(frame, title, focus_on_high=False):
    if np.array_equal(frame.shape, [1, 10000]) or np.array_equal(frame.shape, [10000, 1]) or np.array_equal(frame.shape, [10000,]):
        new_frame = np.reshape(frame, (100, 100))
    elif not np.array_equal(frame.shape, (100, 100)):
        return
    else:
        new_frame = np.copy(frame)
    plt.figure()
    plt.title(title)
    min_val = np.min(new_frame[new_frame > 0])
    max_val = np.max(new_frame)
    if focus_on_high:
        min_val = (3*min_val + max_val)/4
    sio.savemat('loss_map', {'map':new_frame})
    plt.imshow(new_frame, vmin=min_val, vmax=max_val)
    plt.colorbar()
    plt.show()


def plot_frame_loss(losses, frames):
    plt.figure()
    plt.plot(frames, losses)
    plt.xlabel('frame number')
    plt.ylabel('average loss')
    plt.title('Loss Per Missing Frame')
    plt.show()


def plot_mask(mask):
    bounder = find_boundaries(mask, mode='thin').astype(np.uint8)
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.imshow(bounder, alpha=0.5)
    plt.show()
