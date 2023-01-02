import torch


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
