"""
PROJECT MAIN

***Input arguments***:
filename: a relative path to the data file
flags: the type of data to start with. options: raw, seg, set and res.
frames: the relevant frames to take from a raw data input.

***Program order***:
1. READ THE DATA FILE
    a. If its read from a matlab file, the format used here contains three mat objects:
    'clean_horiz'
    'clean_vert'
    'clean_blank'
    All of which have a shape of [10,000 X 256 X n_trials],
    where 10,000 is the number of pixels in a frame ([100X100] turned to a 1d vector) and 256 is the total number of
    frames per trial.
    The 10,000 pixels include pixels that have 0 (or nan) values, and they correspond to points in space that are
    outside the chamber to the brain or that are classified as blood vessels. These pixels were identified during the
    pre-processing of the data and they are ignored in the analysis and model. They are kept in order to recreate the
    [100X100] frames in the end.
    b. The data can be read as numpy files that were created in previous runs in order to save the load time.
2. DIVIDE THE DATA TO SEGMENTS
    a. Choose the relevant frames, and turn the data to [10,000 X n_frames X n_trials].
    Only for the segmentation part: average both conditions over trials to reduce the data to [10,000 X n_frames].
    b. Find a segmentation for the averaged data: a [100X100] mask of integers. Each int indicates a group of
    neighboring pixels that behave similarly over frames.
    This is done using pre-defined hyper parameters.
    c. Transform the data by averaging the pixels of each segment, to get data shape of
    [n_segments X n_frames X n_trials]
3. CREATE THE DATA-SET
    Using the original segmented data from the previous stage augment the examples in the following manner:
    a. For each condition take the average over trials data of shape [n_segments X n_frames]. Let's call it D.
    b. Flatten D to a 1d vector of length [n_segments * n_frames]. We'll call the flatten version D', and its length L.
    c. Find the covariance matrix of the D', let's call it C, such that the shape of C is [L X L].
    d. Take the square root of C - SC, SC*SC=C. The shape of SC is also [L X L].
    e. Find the mean over trials of the segments, we'll call it M, and the shape of M is [L].
    f. Create a random multivariate normal sample of L variables with mean=0 and var=1. We'll call it N.
    g. Generate a new example E: E = N*SC + M. E is reshaped back to [n_segments X n_frames].
    The data set consists of the original examples and the newly created ones.
    While creating the augmented data, a subset of the original data is removed and saved for testing.
4. RUN THE MODEL
    Use the augmented data set and the original examples. Do a 'leave one out' strategy when creating the data - leave
    out an original example and create the data with it, and include this example in the test set.
5. GET RESULTS
    Plot the train and test error as a function of epochs.

"""

import argparse
import data_io
import model
import visualize_res
import segment
import create_data_set
from scipy import io as sio
import dense_net
import data_set
import down_sample

parser = argparse.ArgumentParser()
parser.add_argument("data_folder", help="path to the data file. mat or npz.")
parser.add_argument("-frames", action="store", dest="frames_seg", nargs=2, type=int, default=[27, 53],  # corresponds to 28-54
                    help="a list of integers of the frames to process.")
parser.add_argument("-fr_data", action="store", dest="frames_data", nargs=2, type=int, default=[27, 53],
                    help="a list of integers of the frames to process.")
parser.add_argument("-f", action="store", dest="flag", default="raw", help="Where to start. Options: raw, seg, set, res"
                                                                           "raw: in the beginning."
                                                                           "seg: after segmentation."
                                                                           "set: after creating the data sets."
                                                                           "res: in processing the result")
parser.add_argument("-ds", type=bool, nargs='?', const=True, default=False, help="Down sample frames over time")

args = parser.parse_args()
folder = args.data_folder
frames_seg = list(range(args.frames_seg[0], args.frames_seg[1]))
# frames_data = list(range(args.frames_data[0], args.frames_data[1]))
frames_data = frames_seg
flag = args.flag
should_down_sample = args.ds
cv = True

#################################################################################
# READ THE DATA
print('reading data')
data = data_io.read_from_file(folder, flag)
if flag == "raw":
    v, h = data
elif flag == "seg":
    mask, seg_v, seg_h, frames_data = data
elif flag == "set":
    data_sets = data
elif flag == "los":
    train_losses, validation_losses, test_losses = data
elif flag == "net":
    net = data

#################################################################################
# SEGMENTS
if flag == "raw":
    print('creating segments')
    mask = segment.vert_horiz_seg(v, h, frames_seg)
    if should_down_sample:
        v, _ = down_sample.down_sample_frame(v, frames_data)
        h, frames_data = down_sample.down_sample_frame(h, frames_data)

    seg_v = segment.divide_data_to_segments(mask, v)
    seg_h = segment.divide_data_to_segments(mask, h)
    data_io.save_to([mask, seg_v, seg_h, frames_data], folder, "seg")
#################################################################################
# DATA SET
if flag == "raw" or flag == "seg":
    print('creating data sets')
    data_sets = create_data_set.get_data(
        seg_v, seg_h, n_train=50, n_valid=5, n_test=1, cv=cv, flat_x=True, to_tensor=False, random=False)
    # data_sets contains: train_x, train_y, valid_x, valid_y, test_x, test_y
    data_io.save_to(data_sets, folder, "set")
#################################################################################
# MODEL
if flag == "raw" or flag == "seg" or flag == "set":
    print('running the model')
    train, valid, test, D_in = create_data_set.turn_to_torch_dataset(data_sets, cv=cv)
    train, valid, test = data_set.normalize_datasets([train, valid, test], cv=cv)
    net = dense_net.get_model(D_in)
    # net = dense_net.init_weights(net)
    net, train_losses, validation_losses, test_losses = model.run_model(net, [train, valid, test], cv=cv)
    data_io.save_to([train_losses, validation_losses, test_losses], folder, "los")
    data_io.save_to(net, folder, "net")
#################################################################################

# TODO organize this
net = data_io.read_from_file(folder, "net")
mask, _, _, frames_data = data_io.read_from_file(folder, "seg")
data_sets = data_io.read_from_file(folder, "set")


# train_losses, validation_losses, test_losses = data_io.read_from_file(folder, "los")
# visualize_res.plot_losses(train_losses, validation_losses, test_losses, len(data_sets))

train, valid, test, D_in = create_data_set.turn_to_torch_dataset(data_sets, cv=cv)
train, valid, test = data_set.normalize_datasets([train, valid, test], cv=cv)


loss_map = model.run_with_missing_parts(net, mask, valid, cv, len(frames_data), part_type='frames')
visualize_res.plot_frame_loss(loss_map, [x+1 for x in frames_data])  # counting starts from 0, so the relevant frames are +1
loss_map = model.run_with_missing_parts(net, mask, valid, cv, len(frames_data), part_type='segments')
image = segment.recreate_image(mask, loss_map)
visualize_res.plot_frame(image, "Average Loss for Each Missing Segment")




# sio.savemat('temp_outputs/vis_test', {'image': image})

# image = visualize_res.plot_weights(model_state, mask)
# sio.savemat("temp_outputs/test_vis", {'vis':image})
# visualize_res.plot_losses(train_losses, validation_losses, test_losses)


