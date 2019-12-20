import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):

    def __init__(self, num_classes=2, in_channels=3):
        super(SimpleNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """ forwardという関数にネットワークのforward計算の処理を記述する

        Args:
            x: (torch.Tensor): shapeは(batch_size, num_channels, height, width)

        Returns:
            torch.Tensor: (batch_size, num_classes)の出力
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.avgpool(x)

        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return self.fc(x)


if __name__ == '__main__':
    model = SimpleNet()
    inputs = torch.randn(4, 3, 16, 16)
    outputs = model(inputs)
    # (batch_size, num_classes)の(4, 2)サイズの出力が出る
    print(outputs.size())
