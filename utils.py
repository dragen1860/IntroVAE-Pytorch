import  torch
from    torch import nn



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # print('before flaten:', x.shape)
        return x.view(x.size(0), -1)

class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)

class Add(nn.Module):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, residual):
        """

        :param x:
        :param residual:
        :return:
        """
        return x + residual

    def extra_repr(self):
        return "ResNet Element-wise Add Layer"


class ResBlk(nn.Module):

    def __init__(self, kernels, chs):
        """

        :param kernels: [1, 3, 3]
        :param chs: [ch_in, 64, 64, 64]
        :return:
        """
        super(ResBlk, self).__init__()

        layers = []

        assert len(chs)-1 == len(kernels), "mismatching between chs and kernels"

        # TODO:
        # if K.int_shape(input_tensor[-1]) != filters[0]:

        for idx in range(len(chs)-1):
            layers.extend([
                nn.Conv2d(chs[idx], chs[idx+1], kernel_size=kernels[idx], stride=1, padding=1),
                nn.ReLU(inplace=True)
            ])

        self.net = nn.Sequential(*layers)

        self.shortcut = nn.Sequential()
        if chs[0] != chs[-1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(chs[0], chs[-1], kernel_size=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        res = self.net(x)
        x_ = self.shortcut(x)
        # print(x.shape, x_.shape, res.shape)
        return x_ + res