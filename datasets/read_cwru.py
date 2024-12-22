from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1 encoding
import random
from pathlib import Path
import torch
from torch.utils.data import TensorDataset


# The standard deviation of the training set is used to standardize the training set as well as the test set
def scalar_stand(data_x):
    data_x = data_x.reshape(-1, 1)
    # print(data_x.shape)
    scalar = preprocessing.StandardScaler().fit(data_x)
    data_x = scalar.transform(data_x)
    data_x = data_x.reshape(-1)
    return data_x


# one-hot encoding
def one_hot(data_y):
    data_y = np.array(data_y).reshape([-1, 1])
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(data_y)
    data_y = encoder.transform(data_y).toarray()
    data_y = np.asarray(data_y, dtype=np.int32)
    return data_y


# The build file reads the function capture, which returns the raw data and label data
def capture(original_path):  # Read the MAT file and return a dictionary of attributes
    filenames = os.listdir(original_path)  # Get the payload folder below the 10 file names
    Data_DE = {}
    for i in filenames:  # Traverse through 10 file data
        # File path
        file_path = os.path.join(original_path, i)  # Select the path to a data file
        file = loadmat(file_path)  # dictionary
        file_keys = file.keys()  # All key values of the dictionary
        for key in file_keys:
            if 'DE' in key:  # Get only DE
                Data_DE[i] = file[key].ravel()  # Pull the data into a one-dimensional array
    return Data_DE


# Divide the training sample set and the test sample set
def spilt(data, rate=[0.7, 0.15, 0.15]):  # [[N1],[N2],...,[N10]]
    keys = data.keys()  # 10 file names
    # keys_new = [None for _ in range(len(keys))]
    # for key in keys:  # Sort through the labels in order
    #     if 'normal' in key: keys_new[0] = key
    #     elif 'IR007' in key: keys_new[1] = key
    #     elif 'IR014' in key: keys_new[2] = key
    #     elif 'IR021' in key: keys_new[3] = key
    #     elif 'OR007' in key: keys_new[4] = key
    #     elif 'OR014' in key: keys_new[5] = key
    #     elif 'OR021' in key: keys_new[6] = key
    #     elif 'B007' in key: keys_new[7] = key
    #     elif 'B014' in key: keys_new[8] = key
    #     elif 'B021' in key: keys_new[9] = key
    # keys = keys_new

    tra_data = []
    te_data = []
    val_data = []
    for i in keys:  # Iterate through all folders
        slice_data = data[i]  # Select the data in one file
        slice_data = scalar_stand(slice_data)
        all_length = len(slice_data)  # The length of the data in the file
        tra_data.append(slice_data[0:int(all_length * rate[0])])
        val_data.append((slice_data[int(all_length * rate[0]):int(all_length * (rate[0] + rate[1]))]))
        te_data.append(slice_data[int(all_length * (rate[0] + rate[1])):])
    return tra_data, val_data, te_data


def sampling(data_DE, step=210, sample_len=420):
    sample_DE = []
    label = []
    lab = 0
    for i in range(len(data_DE)):  # Go through 10 files
        all_length = len(data_DE[i])  # The length of the data in the file
        number_sample = int((all_length - sample_len) / step + 1)  # Number of samples
        for j in range(number_sample):  # Sample them one by one
            sample_DE.append(data_DE[i][j * step:j * step + sample_len])
            label.append(lab)
            j += 1
        lab = lab + 1
    return sample_DE, label


def get_data(HP: int, rate, step, sample_len):
    path = str(Path(__file__).resolve().parent / 'data_cwru' / f'{HP}HP')
    data_DE = capture(path)  # Read the data
    train_data_DE, val_data_DE, test_data_DE = spilt(data_DE, rate)  # list [N1,N2,N10]
    x_train, y_train = sampling(train_data_DE, step, sample_len)
    x_validate, y_validate = sampling(val_data_DE, step, sample_len)
    x_test, y_test = sampling(test_data_DE, step, sample_len)
    return np.array(x_train), np.array(y_train), np.array(x_validate), np.array(y_validate), np.array(x_test), np.array(y_test)


def get_data_cwru(src, tar, rate, rate_target, stride, sample_len):

    x_train_source, y_train_source, _, _, _, _ = get_data(src, rate, stride, sample_len)
    x_train_target, y_train_target, _, _, x_test_target, y_test_target = get_data(tar, rate_target, stride, sample_len)

    # Add a dimension to meet the input requirements of conv1D
    # The number of channels will be increased in the later training module
    x_train_source = x_train_source[:, np.newaxis, :]
    x_train_target = x_train_target[:, np.newaxis, :]
    x_test_target = x_test_target[:, np.newaxis, :]

    data_src = torch.tensor(x_train_source, dtype=torch.float)
    data_tar = torch.tensor(x_train_target, dtype=torch.float)
    data_test = torch.tensor(x_test_target, dtype=torch.float)

    train_data_source = TensorDataset(data_src, torch.tensor(y_train_source, dtype=torch.long))
    train_data_target = TensorDataset(data_tar, torch.tensor(y_train_target, dtype=torch.long))
    train_data_test = TensorDataset(data_test, torch.tensor(y_test_target, dtype=torch.long))

    return train_data_source, train_data_target, train_data_test
