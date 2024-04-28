import numpy as np
from sklearn.model_selection import train_test_split
import torch

def get_train_test_sets(data: np.ndarray, sequence_length: int = 5) -> tuple:
    """
    Prepare the training and testing datasets.
    Inputs:
    - data: A numpy array of shape (num_samples,) containing the data.
    - sequence_length: An integer specifying the length of the sequence for the LSTM model.
    Outputs:
    - x_train: A tensor of shape (num_train_samples, sequence_length) containing the training data.
    - x_test: A tensor of shape (num_test_samples, sequence_length) containing the test data.
    - y_train: A tensor of shape (num_train_samples,) containing the target values for the training data.
    - y_test: A tensor of shape (num_test_samples,) containing the target values for the test data.
    """
    x, y, = [], []

    for i in range(len(data) - sequence_length - 1):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = torch.tensor(x_train, dtype = torch.float32)
    x_test = torch.tensor(x_test, dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype = torch.float32)
    y_test = torch.tensor(y_test, dtype = torch.float32)

    return x_train, x_test, y_train, y_test