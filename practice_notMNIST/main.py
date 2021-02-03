import argparse

import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchsummary import summary


from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config # from config.py
from network import Model

SEED = 42
torch.manual_seed(SEED)


def get_config():
    parser = argparse.ArgumentParser(description='Multi-layer perceptron')
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=.001, type=float)
    
    args = parser.parse_args()

    config = Config(
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )

    return config


def get_data(BATCH_SIZE: int):
    print(f'Getting Data...')
    FILE_PATH = '/content/drive/MyDrive/Colab Notebooks/notMNIST_small'
    
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    data = ImageFolder(FILE_PATH, transform=transform)
    
    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size

    train_data, test_data = random_split(data, [train_size, test_size])
    train_iter = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_iter = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    return train_iter, test_iter

def get_network(LEARNING_RATE: float, device: str):
    network = Model().to(device) # model
    criterion = nn.CrossEntropyLoss() # loss
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    return network, criterion, optimizer

def print_modelinfo(model: nn.Module):
    total_params = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad: # [?]
            total_params += len(param.reshape(-1))
    print(f'Number of Total Parameters: {total_params:,d}')


def display_network_summary(network: nn.Module, input_size: tuple):
    summary(network, input_size)


def test_eval(model: nn.Module, test_iter, batch_size: int, device: str):
    with torch.no_grad():
        test_loss = 0
        total = 0
        correct = 0

        for batch_img, batch_lab in test_iter:
            X = batch_img.view(-1, 28*28).to(device)
            Y = batch_lab.to(device)
            y_pred = model.forward(X)
            _, predicted = torch.max(y_pred.data, 1)
            correct += (predicted == Y).sum().item()
            total += batch_img.size(0)
        val_acc = 100 * correct / total
    return val_acc


def train_model(
    model: nn.Module, train_iter, test_iter, EPOCHS: int, BATCH_SIZE: int, device: str
):
    
    # train phase
    print_every = 1
    print(f'Start training! batch size: {BATCH_SIZE}')

    model.train()
    for epoch in range(EPOCHS):
        loss_val_sum = 0
        for batch_img, batch_lab in tqdm(train_iter):
            X = batch_img.view(-1, 28*28).to(device)
            Y = batch_lab.view(BATCH_SIZE).to(device)
            
            y_pred = model.forward(X)
            loss = criterion(y_pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val_sum += loss

        if ((epoch % print_every) == 0) or (epoch == (EPOCHS -1)):
            loss_val_avg = loss_val_sum / len(train_iter)
            accr_val = test_eval(model, test_iter, BATCH_SIZE, device)
            print(
                f"epoch:[{epoch+1}/{EPOCHS}] cost:[{loss_val_avg:.3f}] test_accuracy:[{accr_val:.3f}]"
            )
    print('Training Done!')


def test_model(model, test_iter, device: str):
    raise NotImplementedError()
    # model.eval() # test phase
    # mnist_test = test_iter.datasets
    
    # n_sample = 64
    # sample_indices = np.random.choice(len(mnist.targets), n_sample, replace=False)
    # test_x = mnist_test.data[sample_indices]
    # test_y = mnist_test.targets[sample_indices]

    # with torch.no_grad():
    #     y_pred = model.forward(test_x.view(-1, 28*28).type(torch.float).to(device))
    
    # y_pred = y_pred.argmax(axis=1)

    # plt.figure(figsize=(20, 20))

    # for idx in range(n_sample):
    #     plt.subplot(8, 8, idx + 1)
    #     plt.imshow(test_x[idx], cmap='gray')
    #     plt.axis('off')
    #     plt.title(f'Predict: {y_pred[idx]}, Label: {test_y[idx]}')
    
    # plt.show()

if __name__ == '__main__':
    print(f'PyTorch version: {torch.__version__}')

    config = get_config()
    print(f'This code use {config.device}. Batch size: {config.BATCH_SIZE}')

    train_iter, test_iter = get_data(config.BATCH_SIZE)
    print(f'Preparing dataset done! Batch size: {train_iter.batch_size, test_iter.batch_size}')

    network, criterion, optimizer = get_network(config.LEARNING_RATE, config.device)
    display_network_summary(network, (784, ))
    # print_modelinfo(network)

    train_model(
        network, train_iter, test_iter, config.EPOCHS, config.BATCH_SIZE, config.device
    )

    # test_model(network, test_iter, config.device)
