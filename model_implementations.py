import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, List, Optional, Union, Type

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """
        Dependency for PyTorch ResNet class
        3x3 convolution with padding
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
        Dependency for PyTorch ResNet class
        1x1 convolution
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    
    """
        Dependency for PyTorch ResNet class
    """
    
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    
    """
        Modified version of PyTorch ResNet implemntaion
        Included for experimentation purposes.
    """
    
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 2,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                nn.Dropout2d(p = 0.1)
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
class ResidualAttentionModel(nn.Module):

    """
        Implementation of residual attention model inline with 
        https://doi.org/10.48550/arXiv.1704.06904
        
        This code in based on https://github.com/Necas209/ResidualAttentionNetwork-PyTorch
    """
    
    def __init__(self, num_classes = 7):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 128)
        self.dropout2d1 = nn.Dropout2d(p = 0.2)
        self.attention_module1 = AttentionModule(128, 128)
        self.residual_block2 = ResidualBlock(128, 256, 2)
        self.dropout2d2 = nn.Dropout2d(p = 0.2)
        self.attention_module2 = AttentionModule(256, 256)
        self.residual_block3 = ResidualBlock(256, 256, 2)
        self.dropout2d3 = nn.Dropout2d(p = 0.2)
        self.avgpool = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(p = 0.1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        
        out = self.residual_block1(out)
        out = self.dropout2d1(out)
        
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.dropout2d2(out)
        
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        out = self.dropout2d3(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)

        return out

class ResidualBlock(nn.Module):
    
    """
        Part of implementation of residual attention model inline with 
        https://doi.org/10.48550/arXiv.1704.06904
        
        Code is based on: https://github.com/Necas209/ResidualAttentionNetwork-PyTorch
    """
    
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels // 4, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels // 4, output_channels // 4, 3, stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels // 4, output_channels, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out

class AttentionModule(nn.Module):
    
    """
        Part of implementation of residual attention model inline with 
        https://doi.org/10.48550/arXiv.1704.06904
        
        Code is based on: https://github.com/Necas209/ResidualAttentionNetwork-PyTorch
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)
        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.softmax4_blocks = ResidualBlock(in_channels, out_channels)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)

        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)

        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)

        out_interp3 = F.interpolate(out_softmax3, size=out_softmax2.shape[2:], mode='bilinear', align_corners=False)
        out = out_interp3 + out_softmax2 + out_skip2_connection

        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = F.interpolate(out_softmax4, size=out_softmax1.shape[2:], mode='bilinear', align_corners=False)
        out = out_interp2 + out_softmax1 + out_skip1_connection

        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = F.interpolate(out_softmax5, size=out_trunk.shape[2:], mode='bilinear', align_corners=False)
        out_softmax6 = self.softmax6_blocks(out_interp1)

        out = (1 + out_softmax6) * out_trunk
        out_last = self.last_blocks(out)

        return out_last

class ChainedModel(nn.Module):
    
    """
        Implementation of model that allows chaining of a binary classifier and
        a multi-label classifier into a single model.
    """
    
    def __init__(self, binary_model, multi_model):
        super().__init__()
        self.binary = binary_model  # Outputs logits of shape [batch, 2]
        self.multi = multi_model    # Outputs logits of shape [batch, num_classes]

    def forward(self, x):
        binary_logits = self.binary(x)                     # [batch, 2]
        binary_pred = torch.argmax(F.softmax(binary_logits, dim=1), dim=1)  # [batch]

        output = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        output[binary_pred == 0] = 0
        
        if (binary_pred == 1).any():
            idx = (binary_pred == 1)
            multi_logits = self.multi(x[idx])                # [sub_batch, num_classes]
            multi_pred = torch.argmax(multi_logits, dim=1) + 1  # [sub_batch]
            output[idx] = multi_pred
            
        return output  # [batch]