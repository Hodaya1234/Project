import numpy as np
from matplotlib import pyplot as plt
import segment


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


def plot_frame(frame, title):
    if np.array_equal(frame.shape, [1, 10000]) or np.array_equal(frame.shape, [10000, 1]) or np.array_equal(frame.shape, [10000,]):
        new_frame = np.reshape(frame, (100, 100))
    elif not np.array_equal(frame.shape, (100, 100)):
        return
    else:
        new_frame = np.copy(frame)
    plt.figure()
    plt.title(title)
    plt.imshow(new_frame, vmin=np.min(new_frame[new_frame > 0]), vmax=np.max(new_frame))
    plt.colorbar()
    plt.show()

