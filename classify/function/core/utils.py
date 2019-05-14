import os
import string

import torch
import torch.nn as nn

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_hidden = 128

# Build the category_lines dictionary, a list of names per language

all_categories = [
    "German",
    "English",
    "Czech",
    "Portuguese",
    "Japanese",
    "Polish",
    "Chinese",
    "Scottish",
    "Spanish",
    "Irish",
    "French",
    "Italian",
    "Russian",
    "Vietnamese",
    "Greek",
    "Arabic",
    "Dutch",
    "Korean",
]

n_categories = len(all_categories)

# Just return an output given a line


def evaluate(line_tensor, rnn):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# initialize model schema
rnn = RNN(n_letters, n_hidden, n_categories)


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def letterToIndex(letter):
    return all_letters.find(letter)
