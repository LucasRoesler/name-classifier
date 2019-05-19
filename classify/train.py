import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from core.const import ALL_CATEGORIES, N_CATEGORIES, N_HIDDEN, N_LETTERS
from core.model import RNN
from core.utils import line_to_tensor, load_categories, random_choice, time_since

N_EPOCHS = 100000
PRINT_EVERY = 5000
PLOT_EVERY = 1000
LEARNING_RATE = (
    0.005
)  # If you set this too high, it might explode. If too low, it might not learn

CATEGORY_LINES = load_categories("data/names/*.txt")


def category_from_output(output):
    _, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return ALL_CATEGORIES[category_i], category_i


def random_training_pair():
    category_choice = random_choice(ALL_CATEGORIES)
    line_choice = random_choice(CATEGORY_LINES[category_choice])
    return (
        category_choice,
        line_choice,
        Variable(torch.LongTensor([ALL_CATEGORIES.index(category_choice)])),
        Variable(line_to_tensor(line_choice)),
    )


rnn = RNN(N_LETTERS, N_HIDDEN, N_CATEGORIES)
optimizer = torch.optim.SGD(rnn.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data.item()


if __name__ == "__main__":
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        category, line, category_tensor, line_tensor = random_training_pair()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % PRINT_EVERY == 0:
            guess, guess_i = category_from_output(output)
            correct = "✓" if guess == category else "✗ (%s)" % category
            print(
                "%d %d%% (%s) %.4f %s / %s %s"
                % (
                    epoch,
                    epoch / N_EPOCHS * 100,
                    time_since(start),
                    loss,
                    line,
                    guess,
                    correct,
                )
            )

        # Add current loss avg to list of losses
        if epoch % PLOT_EVERY == 0:
            all_losses.append(current_loss / PLOT_EVERY)
            current_loss = 0

    torch.save(rnn.state_dict(), "data/char-rnn-classification.pt")
