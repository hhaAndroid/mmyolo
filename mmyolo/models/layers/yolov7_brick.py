import torch.nn as nn
import torch
import numpy as np
import math


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=0.03, eps=0.001)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1, momentum=0.03, eps=0.001) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, momentum=0.03, eps=0.001),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, momentum=0.03, eps=0.001),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")

        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

            # print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        # print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        # print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def parse_model(backbone, head, depth_multiple=1.0, width_multiple=1.0, ch=[3], nc=80):  # model_dict, input_channels(3)
    nc, gd, gw = nc, depth_multiple, width_multiple
    na = 3
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], 3  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(backbone + head):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RepConv, SPPCSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return layers, save


v7l_backbone = [[-1, 1, Conv, [32, 3, 1]],  # 0

                [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
                [-1, 1, Conv, [64, 3, 1]],

                [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4
                [-1, 1, Conv, [64, 1, 1]],
                [-2, 1, Conv, [64, 1, 1]],
                [-1, 1, Conv, [64, 3, 1]],
                [-1, 1, Conv, [64, 3, 1]],
                [-1, 1, Conv, [64, 3, 1]],
                [-1, 1, Conv, [64, 3, 1]],
                [[-1, -3, -5, -6], 1, Concat, [1]],
                [-1, 1, Conv, [256, 1, 1]],  # 11

                [-1, 1, MP, []],
                [-1, 1, Conv, [128, 1, 1]],
                [-3, 1, Conv, [128, 1, 1]],
                [-1, 1, Conv, [128, 3, 2]],
                [[-1, -3], 1, Concat, [1]],  # 16-P3/8
                [-1, 1, Conv, [128, 1, 1]],
                [-2, 1, Conv, [128, 1, 1]],
                [-1, 1, Conv, [128, 3, 1]],
                [-1, 1, Conv, [128, 3, 1]],
                [-1, 1, Conv, [128, 3, 1]],
                [-1, 1, Conv, [128, 3, 1]],
                [[-1, -3, -5, -6], 1, Concat, [1]],
                [-1, 1, Conv, [512, 1, 1]],  # 24

                [-1, 1, MP, []],
                [-1, 1, Conv, [256, 1, 1]],
                [-3, 1, Conv, [256, 1, 1]],
                [-1, 1, Conv, [256, 3, 2]],
                [[-1, -3], 1, Concat, [1]],  # 29-P4/16
                [-1, 1, Conv, [256, 1, 1]],
                [-2, 1, Conv, [256, 1, 1]],
                [-1, 1, Conv, [256, 3, 1]],
                [-1, 1, Conv, [256, 3, 1]],
                [-1, 1, Conv, [256, 3, 1]],
                [-1, 1, Conv, [256, 3, 1]],
                [[-1, -3, -5, -6], 1, Concat, [1]],
                [-1, 1, Conv, [1024, 1, 1]],  # 37

                [-1, 1, MP, []],
                [-1, 1, Conv, [512, 1, 1]],
                [-3, 1, Conv, [512, 1, 1]],
                [-1, 1, Conv, [512, 3, 2]],
                [[-1, -3], 1, Concat, [1]],  # 42-P5/32
                [-1, 1, Conv, [256, 1, 1]],
                [-2, 1, Conv, [256, 1, 1]],
                [-1, 1, Conv, [256, 3, 1]],
                [-1, 1, Conv, [256, 3, 1]],
                [-1, 1, Conv, [256, 3, 1]],
                [-1, 1, Conv, [256, 3, 1]],
                [[-1, -3, -5, -6], 1, Concat, [1]],
                [-1, 1, Conv, [1024, 1, 1]]  # 50
                ]

v7l_head = [[-1, 1, SPPCSPC, [512]],  # 51

            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [37, 1, Conv, [256, 1, 1]],  # route backbone P4
            [[-1, -2], 1, Concat, [1]],

            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],  # 63

            [-1, 1, Conv, [128, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [24, 1, Conv, [128, 1, 1]],  # route backbone P3
            [[-1, -2], 1, Concat, [1]],

            [-1, 1, Conv, [128, 1, 1]],
            [-2, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [128, 1, 1]],  # 75

            [-1, 1, MP, []],
            [-1, 1, Conv, [128, 1, 1]],
            [-3, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [128, 3, 2]],
            [[-1, -3, 63], 1, Concat, [1]],

            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],  # 88

            [-1, 1, MP, []],
            [-1, 1, Conv, [256, 1, 1]],
            [-3, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, -3, 51], 1, Concat, [1]],

            [-1, 1, Conv, [512, 1, 1]],
            [-2, 1, Conv, [512, 1, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [512, 1, 1]],  # 101

            [75, 1, RepConv, [256, 3, 1]],
            [88, 1, RepConv, [512, 3, 1]],
            [101, 1, RepConv, [1024, 3, 1]]]
