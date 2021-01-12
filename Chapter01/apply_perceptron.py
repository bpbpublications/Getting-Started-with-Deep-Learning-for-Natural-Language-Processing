# -*- coding: utf-8 -*-
"""
## Author: Sunil Patel
## Copyright: Copyright 2018-2019, Packt Publishing Limited
## Version: 0.0.1
## Maintainer: Sunil Patel
## Email: snlpatel01213@hotmail.com
## Linkedin: https://www.linkedin.com/in/linus1/
## Contributor : {if you debug, append your name here}
## Contributor Email : {if you debug, append your email here}
## Status: active
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

torch.manual_seed(999)

if torch.cuda.is_available(): torch.cuda.manual_seed_all(999)


class data_generator:
    """
    Generating Positive and Negative Examples
    """

    def make_positive(self, mask):
        """
        making positive example
        :param mask:
        :return:
        """
        mask[5, 4] = 1
        mask[4, 5] = 1
        mask[5, 6] = 1
        mask[6, 5] = 1
        mask[3, 3] = 0
        mask[3, 8] = 0
        mask[8, 3] = 0
        mask[8, 8] = 0
        return mask

    def make_negative(self, mask):
        """
        making negative example
        :param mask:
        :return:
        """
        mask[3, 3] = 1
        mask[3, 8] = 1
        mask[8, 3] = 1
        mask[8, 8] = 1
        mask[5, 4] = 0
        mask[4, 5] = 0
        mask[5, 6] = 0
        mask[6, 5] = 0
        return mask

    def generator(self, batch_size):
        """
        generator yields upon being called
        :param batch_size:
        :return:
        """
        data_batch = []
        data_label = []
        for i in range(batch_size):
            random_grid = np.random.random([10, 10])
            mask = random_grid
            data_batch.append(self.make_positive(mask).reshape(10 * 10))
            data_label.append([0, 1])
            random_grid = np.random.random([10, 10])
            mask = random_grid
            data_batch.append(self.make_negative(mask).reshape(10 * 10))
            data_label.append([1, 0])
        yield np.array(data_batch, dtype=np.float16), np.array(data_label, dtype=np.float16)


def accuracy_counter(predicted, target):
    """
    Accuracy measurement
    :param predicted:
    :param target:
    :return:
    """
    predicted_max = predicted.argmax(dim=1)
    target_max = torch.Tensor(target).argmax(dim=1)
    return sum(predicted_max == target_max) / 10.0


class simple_module_1(nn.Module):
    """
    PyTorch model
    """

    def __init__(self):
        super(simple_module_1, self).__init__()
        self.simple_linear = nn.Linear(100, 2)

    def forward(self, input):
        return self.simple_linear(input)


def Train(model, data_generator_obj, optimizer, epochs=1000):
    """
    Training mo
    :param model:
    :param data_generator_obj:
    :param optimizer:
    :param epochs:
    :return:
    """
    optimizer.zero_grad()
    loss_tracker = []
    Accuracy_tracker = []
    for epoch in range(epochs):
        loss = 0
        for x, y in data_generator_obj.generator(5):
            y_ = model(torch.Tensor(x))
            Accuracy = accuracy_counter(y_, y)
            loss += objective(y_, torch.Tensor(y).float())
            loss_tracker.append(loss)
            Accuracy_tracker.append(Accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_tracker, Accuracy_tracker


if __name__ == '__main__':
    # Making object of the class
    simple_module = simple_module_1()
    # define loss
    objective = nn.MSELoss(reduce=True)
    # define Optimizer
    optimizer = optim.SGD(simple_module.parameters(), lr=0.01)
    DG = data_generator()
    loss_1, accuracy = Train(simple_module, DG, optimizer, epochs=200)
    # plotting
    plt.plot(loss_1, label='Loss')
    plt.plot(accuracy, label='Accuracy')
    plt.ylabel("Loss / Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()
