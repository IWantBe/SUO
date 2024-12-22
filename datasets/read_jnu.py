import numpy as np
import os
from sklearn import preprocessing  # 0-1 encoding
import random
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import TensorDataset


# The standard deviation of the training set is used to standardize the training set as well as the test set
def scalar_stand(data_x):
    scalar = preprocessing.StandardScaler().fit(data_x)
    data_x = scalar.transform(data_x)
    data_x = data_x.reshape(-1)
    return data_x


# The build file reads the function capture, which returns the raw data and label data
def capture(original_path):  # Read the MAT file and return a dictionary of attributes
    filenames = os.listdir(original_path)  # Get the payload folder below the 10 file names
    Data_use = []
    for i in filenames:  # Traverse through 10 file data
        # File path
        file_path = os.path.join(original_path, i)  # Select the path to a data file
        file = pd.read_csv(file_path)  # array
        Data_use.append(file)
    return Data_use


# Divide the training sample set and the test sample set
def spilt(data, rate):  # [[N1],[N2],...,[N10]]
    tra_data = []
    te_data = []
    val_data = []
    for i in range(len(data)):  # Iterate through all folders
        slice_data = data[i]  # Select the data in one file
        slice_data = scalar_stand(slice_data)
        all_length = len(slice_data)  # The length of the data in the file
        tra = np.array(slice_data[0:int(all_length * rate[0])]).flatten()
        tra_data.append(tra)
        val = np.array(slice_data[int(all_length * rate[0]):int(all_length * (rate[0] + rate[1]))]).flatten()
        val_data.append(val)
        tes = np.array(slice_data[int(all_length * (rate[0] + rate[1])):]).flatten()
        te_data.append(tes)
        # Row and column conversion

    return tra_data, val_data, te_data


def sampling(data, stride, sample_len):
    sample = []
    label = []
    for i in range(len(data)):  # Go through 10 files
        all_length = len(data[i])  # The length of the data in the file
        number_sample = int((all_length - sample_len) / stride + 1)  # Number of samples
        for j in range(number_sample):  # Sample them one by one
            sample.append(data[i][j * stride:j * stride + sample_len])
            label.append(i)
            j += 1
    return sample, label


def get_data(HP, rate, stride, sample_len):
    path = str(Path(__file__).resolve().parent / 'data_jnu' / f'{HP}')
    data = capture(path)  # read data
    train_data, val_data, test_data = spilt(data, rate)  # list [N1,N2,N10]
    x_train, y_train = sampling(train_data, stride, sample_len)
    x_validate, y_validate = sampling(val_data, stride, sample_len)
    x_test, y_test = sampling(test_data, stride, sample_len)
    return np.array(x_train), np.array(y_train), np.array(x_validate), np.array(y_validate), np.array(x_test), np.array(y_test)


def get_data_jnu(src, tar, rate, rate_target, stride, sample_len):

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
