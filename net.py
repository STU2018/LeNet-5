from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.Sigmoid = nn.Sigmoid()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
                            padding=0)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=10)

    def forward(self, img):
        output = self.c1(img)
        output = self.Sigmoid(output)
        output = self.s2(output)
        output = self.c3(output)
        output = self.Sigmoid(output)
        output = self.s4(output)
        output = self.c5(output)
        output = self.flatten(output)
        output = self.f6(output)
        output = self.output(output)
        return output
