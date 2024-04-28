import torch
import torch.nn as nn
from torch.optim import Optimizer

class LSTM(nn.Module):
    def __init__(self, input_size: int = 5, hidden_layer_size: int = 250, output_size: int = 1):
        """
        Initialize the LSTM model.
        Inputs:
        - input_size: The number of input features.
        - hidden_layer_size: The number of hidden units in the LSTM layer.
        - output_size: The number of output features.
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        Input:
        - input_seq: A tensor of shape (sequence_length, input_size) containing the input sequence.
        Output:
        - A tensor of shape (sequence_length, output_size) containing the model's predictions.
        """
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions
    
    def train_model(self, x_train: torch.Tensor, y_train: torch.Tensor, loss_function: nn.Module, optimizer: Optimizer, epochs: int):
        """
        Train the LSTM model.
        Inputs:
        - x_train: A tensor of shape (num_samples, sequence_length, input_size) containing the training data.
        - y_train: A tensor of shape (num_samples, output_size) containing the target values for the training data.
        - loss_function: The loss function to use for training.
        - optimizer: The optimizer to use for training.
        - epochs: The number of epochs to train for.
        """
        for i in range(epochs):
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                                torch.zeros(1, 1, self.hidden_layer_size))

            y_pred = self(x_train)
            single_loss = loss_function(y_pred, y_train)
            single_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_model(self, x_test: torch.Tensor) -> torch.Tensor:
        """
        Test the LSTM model.
        Input:
        - x_test: A tensor of shape (num_samples, sequence_length, input_size) containing the test data.
        Output:
        - A tensor of shape (num_samples, output_size) containing the model's predictions for the test data.
        """
        self.eval()
        with torch.no_grad():
            y_test_pred = self(x_test)
        return y_test_pred
    
    def save_model(self, file_path: str):
        """
        Save the LSTM model to a pickle file.
        Input:
        - file_path: A string specifying the path to the file where the model should be saved.
        """
        if not file_path.endswith('.pkl'):
            file_path += '.pkl'
            
        torch.save(self.state_dict(), file_path)