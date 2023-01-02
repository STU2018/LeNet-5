from torch.utils.data import Dataset
import os
import cv2


class own_test_dataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data = self.imgs[idx]
        return data


def deal_test_imgs():
    path = 'data/test_pic/hand_write'
    file_list = os.listdir(path)

    file_inx = 0
    for file in file_list:
        img = cv2.imread(path + '/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))

        for i in range(28):
            for j in range(28):
                if img[i][j] < 128:
                    img[i][j] = 0
                else:
                    img[i][j] = 255

        cv2.imwrite(path + '/' + file, img)

    for file in file_list:
        front, end = file.split('.')
        front = str(file_inx)
        front = front.zfill(2)
        new_name = '.'.join([front, end])
        os.rename(path + '/' + file, path + '/' + new_name)
        file_inx += 1
