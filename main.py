import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os
import argparse
from torch.autograd import Variable
from net import LeNet5


def train(epoch, dataloader, model, loss_fn, optimizer, device):
    # switch to train mode
    model.train()

    train_loss, train_acc, batch_num = 0.0, 0.0, 0
    for batch_index, (X, y) in enumerate(dataloader):
        # transfer to target device
        X, y = X.to(device), y.to(device)

        # forward
        output = model(X)

        # calculate loss
        batch_loss = loss_fn(output, y)

        # calculate acc
        _, pred = torch.max(output, axis=1)
        batch_acc = torch.sum(y == pred) / output.shape[0]

        # backward: calculate grad and update parameters
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # sum the loss and acc
        train_loss += batch_loss.item()
        train_acc += batch_acc.item()
        batch_num = batch_num + 1

    print('epoch: {}, loss: {}, train acc: {:.2f}%'.format(epoch + 1, train_loss / batch_num,
                                                           100 * train_acc / batch_num))
    return train_acc / batch_num


def test(dataset, model, device, model_save_path="save/best_model.pth", test_result_save_path="save/test_results.txt"):
    # switch to eval mode
    model.eval()
    model = model.to(device)

    # load model
    model.load_state_dict(torch.load(model_save_path))

    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    save_file = open(test_result_save_path, "w")

    test_acc = 0.0
    for test_data in dataset:
        X, y = test_data[0], test_data[1]

        X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False).to(device)
        with torch.no_grad():
            ret = model(X)

            predict, real = classes[torch.argmax(ret[0])], classes[y]
            save_file.write(
                'predict = {} , real = {}, {} \n'.format(predict, real, 'right' if predict == real else 'wrong'))
            if predict == real:
                test_acc = test_acc + 1.0

    save_file.close()

    print("test done \ntest acc: {:.2f}%".format(100 * test_acc / len(dataset)))


def main():
    # argument setting
    parser = argparse.ArgumentParser(description='PyTorch Implementation of LeNet-5')
    parser.add_argument("--mode", type=str, default='train', help="train LeNet-5 model")

    # transform to tensor
    data_transform = transforms.Compose([
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
        train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
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
        test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)

        print("start to test model")
        test(test_dataset, model, device)

    # argument wrong
    else:
        print("argument wrong!")


if __name__ == '__main__':
    main()
