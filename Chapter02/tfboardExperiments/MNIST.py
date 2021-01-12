# -*- coding: utf-8 -*-
"""
## Author: Sunil Patel
## Copyright: Copyright 2018-2019, Packt Publishing Limited
## Version: 0.0.1
## Maintainer: sunil patel
## Email: snlpatel01213@hotmail.com
## Linkedin: https://www.linkedin.com/in/linus1/
## Contributor : {if you debug, append your name here}
## Contributor Email : {if you debug, append your email here}
## Status: active
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torchvision import transforms


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    print(
        """##################################\n## Launch tensorboard as: ## \n## tensorboard --logdir=runs/ ## \n##################################""")
    writer = SummaryWriter()
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST dataset
    dataset = torchvision.datasets.MNIST(root='../../data',
                                         train=True,
                                         transform=transforms.ToTensor(),
                                         download=True)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=100,
                                              shuffle=True)
    model = NeuralNet().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    data_iter = iter(data_loader)
    iter_per_epoch = len(data_loader)
    total_step = 50000

    # Start training
    for step in range(total_step):

        # Reset the data_iter
        if (step + 1) % iter_per_epoch == 0:
            data_iter = iter(data_loader)

        # Fetch images and labels
        images, labels = next(data_iter)
        images, labels = images.view(images.size(0), -1).to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()

        if (step + 1) % 100 == 0:
            print('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
                  .format(step + 1, total_step, loss.item(), accuracy.item()))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log scalar values (scalar summary)
            info = {'loss': loss.item(), 'accuracy': accuracy.item()}

            for tag, value in info.items():
                writer.add_scalar(tag, value, step + 1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag, value.data.cpu().numpy(), step + 1)
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

            # 3. Log training images (image summary)
            info = {'images': images.view(-1, 1, 28, 28)[:10].cpu()}

            for tag, images in info.items():
                x = vutils.make_grid(images, normalize=True, scale_each=True)
                writer.add_image(tag, x, step + 1)
