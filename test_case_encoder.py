import torch
import torch.nn as nn
"""
Test Case Encoder

This encoder uses two separate BiLSTM networks to encode the input 
and output sequences of the test case. The final hidden states from 
the input and output encoders are concatenated, and a linear projection 
is applied to obtain the test case embedding.
"""
class TestCaseEncoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(TestCaseEncoder, self).__init__()
        self.input_encoder = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.output_encoder = nn.LSTM(output_size, hidden_size, bidirectional=True, batch_first=True)
        self.projection = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, input_seq, output_seq):
        input_output, _ = self.input_encoder(input_seq)
        output_output, _ = self.output_encoder(output_seq)

        input_repr = input_output[:, -1, :]  # Take the last hidden state from the input encoder
        output_repr = output_output[:, -1, :]  # Take the last hidden state from the output encoder

        concat_repr = torch.cat((input_repr, output_repr), dim=1)  # Concatenate input and output representations
        projected_output = self.projection(concat_repr)
        return projected_output