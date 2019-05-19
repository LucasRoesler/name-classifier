import json
import os
from pathlib import PurePath
from typing import Any, List

import torch
from torch.autograd import Variable

from .core import const, model, utils

FUNCTION_ROOT = os.environ.get("function_root", "/root/function/")

# init model
RNN = model.RNN(const.N_LETTERS, const.N_HIDDEN, const.N_CATEGORIES)
# fill in weights
RNN.load_state_dict(
    torch.load(str(PurePath(FUNCTION_ROOT, "data/char-rnn-classification.pt")))
)


def predict(line: str, n_predictions: int = 3) -> List[Any]:
    output = model.evaluate(Variable(utils.line_to_tensor(line)), RNN)

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions: List[Any] = []

    for i in range(n_predictions):
        value = str(topv[0][i]).split("tensor")[1]
        category_index = topi[0][i]
        predictions += [(value, const.ALL_CATEGORIES[category_index])]

    return predictions


def handle(req: bytes) -> str:
    """handle a request to the function
    Args:
        req (bytes): request body
    """

    if not req:
        return json.dumps({"error": "No input provided", "code": 400})

    name = str(req)
    output = predict(name)

    return json.dumps(output)
