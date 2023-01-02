import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os
import argparse

from net import LeNet5
from train import train
from test import test
from utils import deal_test_imgs, own_test_dataset


def main():
    # argument setting
    parser = argparse.ArgumentParser(description='PyTorch Implementation of LeNet-5')
    parser.add_argument("--mode", type=str, default='train',
                        help="train: train LeNet-5 model test:  test LeNet-5 model")
    parser.add_argument("--test_mode", type=str, default='mnist',
                        help="mnist: test on mnist dataset ,custom: test on your own dataset")

    # transform to tensor
    data_transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor()
    ])

    # device setting
    device = "cuda" if torch.cuda.is_available() == True else 'cpu'

    # define model and transfer to target decvice
    model = LeNet5()
    model = model.to(device)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    # train model
    if parser.parse_args().mode == 'train':
        # load train dataset
        train_dataset = datasets.MNIST(root='./data',
                                       train=True,
                                       transform=transforms.Compose([
                                           transforms.Resize([32, 32]),
                                           transforms.ToTensor()
                                       ])
                                       , download=True)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

        # define optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        print("start to train model")
        epoch = 50
        best_train_acc = 0
        for _epoch in range(epoch):
            train_acc = train(_epoch, train_dataloader, model, loss_fn, optimizer, device)

            # save best model
            if train_acc > best_train_acc:
                path = 'save'
                if os.path.exists(path) == False:
                    os.mkdir(path)
                best_train_acc = train_acc
                torch.save(model.state_dict(), path + '/best_model.pth')

        print("train done")

    # test model
    elif parser.parse_args().mode == 'test':
        # load test dataset
        if parser.parse_args().test_mode == 'mnist':
            test_dataset = datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.Compose([
                                              transforms.Resize([32, 32]),
                                              transforms.ToTensor()
                                          ])
                                          , download=True)
            print("start to test model")
            test(test_dataset, model, device, use_own_test_imgs=(parser.parse_args().test_mode != 'mnist'))
        else:
            deal_test_imgs()
            test_imgs = ImageFolder(root='./data/test_pic',
                                    transform=transforms.Compose([
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize([32, 32]),
                                        transforms.ToTensor()
                                    ])
                                    )
            test_dataset = own_test_dataset(imgs=test_imgs)
            print("start to test model")
            test(test_dataset, model, device, use_own_test_imgs=(parser.parse_args().test_mode != 'mnist'))



    # argument wrong
    else:
        print("argument wrong!")


if __name__ == '__main__':
    main()
