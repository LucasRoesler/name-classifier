import glob
import math
import random
import time
import unicodedata
from typing import Any, Dict, List

import torch

from .const import ALL_LETTERS, N_LETTERS


def get_secret(secret_name: str) -> str:
    """load secret value from the openfaas secret folder
    Args:
        secret_name (str): name of the secret
    """
    with open("/var/openfaas/secrets/" + secret_name, "r") as file:
        return file.read()


def random_choice(l: List) -> Any:
    return l[random.randint(0, len(l) - 1)]


def time_since(since):
    now = time.time()
    seconds = now - since
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return "%dm %ds" % (minutes, seconds)


def evaluate(line_tensor, rnn):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def line_to_tensor(line):
    """Turn a line into a <line_length x 1 x n_letters>,
    or an array of one-hot letter vectors
    """
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for idx, letter in enumerate(line):
        tensor[idx][0][letter_to_index(letter)] = 1
    return tensor


def letter_to_index(letter: str) -> int:
    """Find letter index from all_letters, e.g. "a" = 0."""
    return ALL_LETTERS.find(letter)


def find_files(path) -> List[str]:
    return glob.glob(path)


def unicode_to_ascii(input_str: str) -> str:
    """Turn a Unicode string to plain ASCII.

    Thanks to http://stackoverflow.com/a/518232/2809427
    """
    return "".join(
        char
        for char in unicodedata.normalize("NFD", input_str)
        if unicodedata.category(char) != "Mn" and char in ALL_LETTERS
    )


def read_lines(filename: str) -> List[str]:
    """Read a file and split into lines."""
    lines = open(filename).read().strip().split("\n")
    return [unicode_to_ascii(line) for line in lines]


def load_categories(file_glob: str) -> Dict[str, List[str]]:
    """Build a list of lines per category"""
    category_lines = {}
    for filename in find_files(file_glob):
        category_name = filename.split("/")[-1].split(".")[0]
        lines = read_lines(filename)
        category_lines[category_name] = lines
    return category_lines
