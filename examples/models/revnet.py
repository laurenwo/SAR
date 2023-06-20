"""
    RevNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'The Reversible Residual Network: Backpropagation Without Storing Activations,'
    https://arxiv.org/abs/1707.04585.
    Source:
        - https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1
"""

import os
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from .common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, pre_conv1x1_block, pre_conv3x3_block


use_context_mans = int(
    torch.__version__[0]) * 100 + int(torch.__version__[2]) - (1 if 'a' in torch.__version__ else 0) > 3


@contextmanager
def set_grad_enabled(grad_mode):
    if not use_context_mans:
        yield
    else:
        with torch.set_grad_enabled(grad_mode) as c:
            yield [c]


class ReversibleBlockFunction(torch.autograd.Function):
    """
    RevNet reversible block function.
    """

    @staticmethod
    def forward(ctx, x, fm, gm, *params):

        with torch.no_grad():
            x1, x2 = torch.chunk(x, chunks=2, dim=-1)
            x1 = x1.contiguous()
            x2 = x2.contiguous()

            y1 = x1 + fm(x2)
            y2 = x2 + gm(y1)

            y = torch.cat((y1, y2), dim=-1)

            x1.set_()
            x2.set_()
            y1.set_()
            y2.set_()
            del x1, x2, y1, y2

        ctx.save_for_backward(x, y)
        ctx.fm = fm
        ctx.gm = gm

        return y

    @staticmethod
    def backward(ctx, grad_y):
        fm = ctx.fm
        gm = ctx.gm

        x, y = ctx.saved_variables
        y1, y2 = torch.chunk(y, chunks=2, dim=-1)
        y1 = y1.contiguous()
        y2 = y2.contiguous()

        with torch.no_grad():
            y1_z = Variable(y1.data, requires_grad=True)
            x2 = y2 - gm(y1_z)
            x1 = y1 - fm(x2)

        with set_grad_enabled(True):
            x1_ = Variable(x1.data, requires_grad=True)
            x2_ = Variable(x2.data, requires_grad=True)
            y1_ = x1_ + fm.forward(x2_)
            y2_ = x2_ + gm(y1_)
            y = torch.cat((y1_, y2_), dim=-1)

            dd = torch.autograd.grad(y, (x1_, x2_) + tuple(gm.parameters()) + tuple(fm.parameters()), grad_y)

            gm_params_len = len([p for p in gm.parameters()])
            gm_params_grads = dd[2:2 + gm_params_len]
            fm_params_grads = dd[2 + gm_params_len:]
            grad_x = torch.cat((dd[0], dd[1]), dim=-1)

            y1_.detach_()
            y2_.detach_()
            del y1_, y2_

        x.data.set_(torch.cat((x1, x2), dim=-1).data.contiguous())

        return (grad_x, None, None) + fm_params_grads + gm_params_grads


class RevResBlock(nn.Module):
    """
    Simple RevNet block for residual path in RevNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    preactivate : bool
        Whether use pre-activation for the first convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 preactivate):
        super(RevResBlock, self).__init__()
        if preactivate:
            self.conv1 = pre_conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        else:
            self.conv1 = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        self.conv2 = pre_conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RevNet(nn.Module):
    """
    RevNet model from 'The Reversible Residual Network: Backpropagation Without Storing Activations,'
    https://arxiv.org/abs/1707.04585.

    Parameters:
    ----------
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    """
    def __init__(self,
                 in_size=(128,)):
        super(RevNet, self).__init__()
        self.in_size = in_size

        self.gm = RevResBlock(
            in_channels=1,
            out_channels=1,
            stride=1,
            preactivate=False)
        self.fm = RevResBlock(
            in_channels=1,
            out_channels=1,
            stride=1,
            preactivate=False)
        self.rev_funct = ReversibleBlockFunction.apply

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        assert (x.shape[-1] % 2 == 0)

        params = [w for w in self.fm.parameters()] + [w for w in self.gm.parameters()]
        y = self.rev_funct(x, self.fm, self.gm, *params)

        x.data.set_()

        return y

    def inverse(self, y):
        assert (y.shape[-1] % 2 == 0)

        y1, y2 = torch.chunk(y, chunks=2, dim=1)
        y1 = y1.contiguous()
        y2 = y2.contiguous()

        x2 = y2 - self.gm(y1)
        x1 = y1 - self.fm(x2)

        x = torch.cat((x1, x2), dim=1)
        return x


