"""
PROJECT MAIN

***Input arguments***:
filename: a relative path to the data file
flags: the type of data to start with. options: raw, seg, set and res.

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
    visualize the network by removing parts and seeing the effect on the loss

"""

import argparse

import numpy as np
import model
import create_data_set
import data_io
import data_set
import dense_net
import segment
import visualize_res
from read_settings import Settings


def main(path):
    settings = Settings(path)
    # mask seg set net los vis
    if 'mask' in settings.stages:
        # need v,h raw and frames
        v, h = data_io.read_from_file(settings.files['raw'], 'raw')
        mask = segment.vert_horiz_seg(v[:, settings.frames, :], h[:, settings.frames, :], square=True)
        data_io.save_to(mask, settings.files['mask'], 'mask')

    if 'seg' in settings.stages:
        mask = data_io.read_from_file(settings.files['mask'], 'mask')
        v, h = data_io.read_from_file(settings.files['raw'], 'raw')
        seg_v = segment.divide_data_to_segments(mask, v[:, settings.frames, :])
        seg_h = segment.divide_data_to_segments(mask, h[:, settings.frames, :])
        data_io.save_to([seg_v, seg_h], settings.files['seg'], 'seg')

    if 'set' in settings.stages:
        [seg_v, seg_h] = data_io.read_from_file(settings.files['seg'], 'seg')
        cv = 'cv' in settings.flags
        sizes = settings.sizes
        data_sets = create_data_set.get_data(
            seg_v, seg_h, n_train=sizes['train'], n_valid=sizes['valid'], n_test=sizes['test'], cv=cv, flat_x=True,
            to_tensor=False, random=False)
        data_io.save_to(data_sets, settings.files['set'], 'set')

    if 'net' in settings.stages:
        cv = 'cv' in settings.flags
        zero_all = 'zero_all' in settings.flags
        value_type = 'acc' if 'acc' in settings.flags else 'loss'
        data_sets = data_io.read_from_file(settings.files['set'], 'set')
        mask = data_io.read_from_file(settings.files['mask'], 'mask')

        # TODO: remove this
        train, valid, test, D_in = create_data_set.turn_to_torch_dataset_old(data_sets, cv=cv)

        if not cv:
            net = dense_net.get_model(D_in)
            mean_t, std_t = train.calc_mean_std()
            train, valid, test = train.normalize(mean_t, std_t), valid.normalize(mean_t, std_t), test.normalize(mean_t, std_t)
            training_parameters = model.get_train_params(net)
            net, train_losses, valid_losses, valid_accuracies = model.train(net, [train, test], training_parameters)
            data_io.save_to(net, settings.files['net'], 'net')

        else:
            n_data_sets = len(train)
            n_frames = len(settings.frames)
            mask_nubmers = np.unique(mask)
            n_seg = len(mask_nubmers) - 1 if mask_nubmers[0] == 0 else len(mask_nubmers)

            frames_loss_maps = np.zeros([n_data_sets, n_frames])
            seg_loss_maps = np.zeros([n_data_sets, n_seg])
            all_train_losses = []
            all_valid_losses = []
            for idx, one_train, one_test in zip(range(n_data_sets), train, test):
                mean_t, std_t = one_train.calc_mean_std()
                one_train = one_train.normalize(mean_t, std_t)
                one_test = one_test.normalize(mean_t, std_t)

                net = dense_net.get_model(D_in)
                training_parameters = model.get_train_params(net)

                net, train_losses, valid_losses, valid_accuracies = model.train(net, [one_train, one_test], training_parameters)
                if valid_losses[-1] > 0.6:
                    print('\n{}\n'.format(idx))
                all_train_losses.append(train_losses)
                all_valid_losses.append(valid_losses)
                frames_loss_maps[idx, :] = np.asarray(
                    model.run_with_missing_parts(net, mask, one_test, False, len(settings.frames), part_type='frames',
                                                 zero_all=zero_all, value_type=value_type))
                seg_loss_maps[idx, :] = model.run_with_missing_parts(
                    net, mask, one_test, False, len(settings.frames),
                    part_type='segments', zero_all=zero_all, value_type=value_type)

            frame_loss = np.mean(frames_loss_maps, axis=0)
            seg_loss = segment.recreate_image(mask, np.mean(seg_loss_maps, axis=0))
            data_io.save_to(frame_loss, settings.files['vis_frame'], 'vis')
            data_io.save_to(seg_loss, settings.files['vis_seg'], 'vis')
            visualize_res.plot_losses(all_train_losses, all_valid_losses, [], n_data_sets)

        # n_data_sets = len(data_sets[0]) if cv else 1
        # data_io.save_to([train_losses, validation_losses, test_losses, test_accuracies, n_data_sets], settings.files['los'], 'los')

    if 'los' in settings.stages:
        # TODO: update
        train_losses, validation_losses, test_losses, test_accuracies, n_data_sets = data_io.read_from_file(settings.files['los'], 'los')
        visualize_res.plot_losses(train_losses, validation_losses, test_losses, n_data_sets)

    if 'calc_vis' in settings.stages:
        # TODO: remove
        net = data_io.read_from_file(settings.files['net'], 'net')
        data_sets = data_io.read_from_file(settings.files['set'], 'set')
        mask = data_io.read_from_file(settings.files['mask'], 'mask')
        cv = 'cv' in settings.flags
        zero_all = 'zero_all' in settings.flags
        value_type = 'acc' if 'acc' in settings.flags else 'loss'

        train, valid, test, D_in = create_data_set.turn_to_torch_dataset_old(data_sets, cv=cv)
        train, valid, test = data_set.normalize_datasets([train, valid, test], cv=cv)

        loss_map = model.run_with_missing_parts(net, mask, valid, cv, len(settings.frames), part_type='both', zero_all=zero_all, value_type=value_type)
        loss_map = loss_map.reshape([-1, len(settings.frames)])
        loss_maps = [np.mean(loss_map[:, frames], axis=1) for frames in settings.frame_groups]
        images = np.asarray([segment.recreate_image(mask, one_loss_map) for one_loss_map in loss_maps])
        data_io.save_to(images, settings.files['vis_both'], 'vis')

        loss_map = np.asarray(model.run_with_missing_parts(net, mask, valid, cv, len(settings.frames), part_type='frames', zero_all=zero_all, value_type=value_type))
        data_io.save_to(loss_map, settings.files['vis_frame'], 'vis')

        loss_map = model.run_with_missing_parts(net, mask, valid, cv, len(settings.frames), part_type='segments', zero_all=zero_all, value_type=value_type)
        image = segment.recreate_image(mask, loss_map)
        data_io.save_to(image, settings.files['vis_seg'], 'vis')

    if 'show_vis' in settings.stages:
        zero_all = 'zero_all' in settings.flags
        value_type = 'acc' if 'acc' in settings.flags else 'loss'
        zero_all_str = 'Present' if zero_all else 'Missing'
        value_type_str = 'Accuracy' if value_type == 'acc' else 'Loss'
        title_seg = 'Average {} per {} Segment'.format(value_type_str, zero_all_str)
        title_frame = 'Average {} per {} Frame'.format(value_type_str, zero_all_str)

        # images = data_io.read_from_file(settings.files['vis_both'], 'vis')
        # visualize_res.plot_spatial(images, settings.frame_groups_string, n_frames=len(images))

        loss_map = data_io.read_from_file(settings.files['vis_frame'], 'vis')
        visualize_res.plot_temporal(loss_map, [x + 1 for x in settings.frames], title=title_frame, ylabel=value_type)  # counting starts from 0, so the relevant frames are +1

        image = data_io.read_from_file(settings.files['vis_seg'], 'vis')
        visualize_res.plot_spatial(image, title=title_seg)


parser = argparse.ArgumentParser()
parser.add_argument("settings_path", help="path to the settings file")
args = parser.parse_args()
main(args.settings_path)
