import torch
from torch.autograd import Variable


def test(dataset, model, device, model_save_path="save/best_model.pth", test_result_save_path="save/test_results.txt",
         use_own_test_imgs=False):
    # switch to eval mode
    model.eval()
    model = model.to(device)

    # load model
    model.load_state_dict(torch.load(model_save_path))

    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    save_file = open(test_result_save_path, "w")

    if use_own_test_imgs == True:
        test_inx = -1
        for test_data in dataset:
            test_inx += 1

            X = test_data[0]
            X = Variable(torch.unsqueeze(X, dim=0), requires_grad=False).to(device)
            with torch.no_grad():
                ret = model(X)
                predict = classes[torch.argmax(ret[0])]

                save_file.write('PIC {} : predict = {} \n'.format(test_inx, predict))

    else:
        test_acc = 0.0
        for test_data in dataset:
            X, y = test_data[0], test_data[1]

            X = Variable(torch.unsqueeze(X, dim=0), requires_grad=False).to(device)
            with torch.no_grad():
                ret = model(X)
                predict, real = classes[torch.argmax(ret[0])], classes[y]

                save_file.write(
                    'predict = {} , real = {}, {} \n'.format(predict, real,
                                                             'right' if predict == real else 'wrong'))
                if predict == real:
                    test_acc = test_acc + 1.0

        print("test done \ntest acc: {:.2f}%".format(100 * test_acc / len(dataset)))

    save_file.close()
