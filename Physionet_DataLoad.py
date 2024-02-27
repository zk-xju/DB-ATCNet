import tensorflow as tf
import numpy as np
import os
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.epochs import Epochs
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def standardize_data(X_train, X_test, channels):
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

def to_one_hot(y, by_sub=False):
    if by_sub:
        new_array = np.array(["nan" for nan in range(len(y))])
        for index, label in enumerate(y):
            new_array[index] = ''.join([i for i in label if not i.isdigit()])
    else:
        new_array = y.copy()
    total_labels = np.unique(new_array)
    mapping = {}
    for x in range(len(total_labels)):
        mapping[total_labels[x]] = x
    for x in range(len(new_array)):
        new_array[x] = mapping[new_array[x]]

    return tf.keras.utils.to_categorical(new_array)

def load_subject_data(subject: int, data_path: str, exclude_base: bool = False):
    """
    Given a subject number (@subject) and the original dataset
    path (@data_path), this function returns:
        xs: The time series; a numpy array of shape (n_sample, 64, 641)
        y: The labels, a list of length n_samples
        ch_names: The 64 channels order in the xs array
    """
    runs = [4, 6, 8, 10, 12, 14]
    task2 = [4, 8, 12]
    task4 = [6, 10, 14]
    if len(str(subject)) == 1:
        sub_name = "S" + "00" + str(subject)
    elif len(str(subject)) == 2:
        sub_name = "S" + "0" + str(subject)
    else:
        sub_name = "S" + str(subject)
    sub_folder = os.path.join(data_path, sub_name)
    subject_runs = []
    for run in runs:
        if len(str(run)) == 1:
            path_run = os.path.join(sub_folder,
                                    sub_name + "R" + "0" + str(run) + ".edf")
        else:
            path_run = os.path.join(sub_folder,
                                    sub_name + "R" + str(run) + ".edf")
        raw_run = read_raw_edf(path_run, preload=True)
        len_run = np.sum(
            raw_run._annotations.duration)
        if len_run > 124:
            raw_run.crop(tmax=124)

        """
        B indicates baseline
        L indicates motor imagination of opening and closing left fist;
        R indicates motor imagination of opening and closing right fist;
        LR indicates motor imagination of opening and closing both fists;
        F indicates motor imagination of opening and closing both feet.
        """

        if int(run) in task2:
            for index, an in enumerate(raw_run.annotations.description):
                if an == "T0":
                    raw_run.annotations.description[index] = "B"
                if an == "T1":
                    raw_run.annotations.description[index] = "L"
                if an == "T2":
                    raw_run.annotations.description[index] = "R"
        if int(run) in task4:
            for index, an in enumerate(raw_run.annotations.description):
                if an == "T0":
                    raw_run.annotations.description[index] = "B"
                if an == "T1":
                    raw_run.annotations.description[index] = "LR"
                if an == "T2":
                    raw_run.annotations.description[index] = "F"
        subject_runs.append(raw_run)
    raw_conc = concatenate_raws(subject_runs)
    indexes = []
    for index, value in enumerate(raw_conc.annotations.description):
        if value == "BAD boundary" or value == "EDGE boundary":
            indexes.append(index)
    raw_conc.annotations.delete(indexes)

    eegbci.standardize(raw_conc)
    montage = make_standard_montage('standard_1005')
    raw_conc.set_montage(montage)
    tmin = 0
    tmax = 4
    if exclude_base:
        event_id = dict(F=2, L=3, LR=4, R=5)#4class
        # event_id = dict(L=3,R=5)#2class

    else:
        event_id = dict(B=1, F=2, L=3, LR=4, R=5)
    events, _ = mne.events_from_annotations(raw_conc, event_id=event_id)

    #64
    picks = mne.pick_types(raw_conc.info, meg=False, eeg=True, stim=False,
                           eog=False, exclude='bads')
    #22
    # picks = mne.pick_types(raw_conc.info, meg=False, eeg=False, stim=False,
    #                        eog=False,include=["Fz", "FC3","FC1", "FCz","FC2", "FC4","C5",  "C3","C1",  "Cz",
    #            "C2",  "C4","C6", "CP3","CP1", "CPz","CP2", "CP4","P1", "Pz","P2","POz"],exclude='bads')

    # 18
    # picks = mne.pick_types(raw_conc.info, meg=False, eeg=False, stim=False,
    #                        eog=False,include=["FC5","FC3","FC1","FC2", "FC4","FC6","C5","C3","C1",
    #            "C2","C4","C6","CP5","CP3","CP1", "CP2", "CP4", "CP6"],exclude='bads')

    #12
    # picks = mne.pick_types(raw_conc.info, meg=False, eeg=False, stim=False,
    #                        eog=False, include=["FC3", "FC1", "FC2", "FC4","C3", "C1","C2", "C4","CP3", "CP1","CP2", "CP4"], exclude='bads')

    epochs = Epochs(raw_conc, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)

    print(epochs[0].ch_names)

    y = list()
    for index, data in enumerate(epochs):
        y.append(epochs[index]._name)

    xs = np.array([epoch for epoch in epochs])

    return xs, y, raw_conc.ch_names

def load_physionet(path):
    exclude = [38, 88, 89, 92, 100, 104]
    subjects = [n for n in np.arange(1, 110) if n not in exclude]

    xs = list()
    ys = list()
    data_x = list()
    data_y = list()
    for subject in subjects:
        x, y, ch_names = load_subject_data(subject,path,True)
        print(x.shape)
        xs.append(x)
        ys.append(y)
    data_x = np.concatenate(xs)
    data_y = np.concatenate(ys)

    N_tr, N_ch, _ = data_x.shape
    data_x = data_x[:, :, :640].reshape(N_tr, 1, N_ch, -1)
    y_one_hot  = to_one_hot(data_y, by_sub=False)

    # Create Validation/test
    x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(data_x,
                                                                                y_one_hot,
                                                                                stratify=y_one_hot,
                                                                                test_size=0.10,
                                                                                random_state=42)

    #Scale indipendently train/test
    x_train_scaled_raw, x_test_valid_scaled_raw = standardize_data(x_train_raw, x_valid_test_raw, N_ch)

    print(x_train_scaled_raw.shape, x_test_valid_scaled_raw.shape)
    return x_train_scaled_raw,y_train_raw,x_test_valid_scaled_raw,y_valid_test_raw

