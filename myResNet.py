import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    """
    Inputs:
        Number of input channel for this BottleNeck block
        Number of output channel from first Conv2d layer in this BottleNeck block
    """
    expansion=4

    def __init__(self, in_ch, out_ch, stride=1):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)     # padding = same = (f-1)/2
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, self.expansion*out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_ch)
        self.relu = nn.ReLU()

        self.need_proj = (stride!=1) or (in_ch!=out_ch*self.expansion)      # need downsample and/or expand
        # stride == 2 -> first bottle neck block in any layer except layer 1 (that have 3 blocks) -> expand channels + downsample H,W
        # in_ch == out_ch in first bottle neck block in layer 1 (that have 3 blocks) and its stride is 1 -> expand only

        if self.need_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, self.expansion*out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.expansion*out_ch)
            )
        else:
            self.proj = nn.Identity()

        nn.init.constant_(self.bn3.weight, 0.0)     # start with res branch -> off, they found it is better

    def forward(self, x):
        identity = self.proj(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels=3):     # layers -> [3, 4, 6, 3] in resnet 50
        super(ResNet, self).__init__()

        self.block = block

        self.conv = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(64, layers[0], 1)               # no stride in first block (no reduction in W,H)
        self.layer2 = self._make_layer(128, layers[1], 2)              # there is Stride = 2 in first bottleneck in each layer
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     # 1x1xChannels
        self.fc = nn.Linear(2048, 1)

    def _make_layer(self, first_conv_channels, number_of_blocks, first_block_stride):

        if first_block_stride==1:
            layers = [self.block(first_conv_channels, first_conv_channels, first_block_stride)]
        else:
            layers = [self.block(first_conv_channels*2, first_conv_channels, first_block_stride)]

        for i in range(number_of_blocks-1):
            layers.append(self.block(first_conv_channels*self.block.expansion, first_conv_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):   # to make each of the four layers each of [3, 4, 6, 3] BottleNeck blocks
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = torch.sigmoid(x)
        return x


def ResNet50(img_channel=3):
    return ResNet(BottleNeck, [3, 4, 6, 3], img_channel)
