import numpy as np
from matplotlib import pyplot as plt
import segment


def plot_losses(train_losses, validation_losses, test_losses):
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(validation_losses, label="validation loss")
    plt.plot(test_losses, label="test loss")
    plt.legend()
    plt.savefig('losses')
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


