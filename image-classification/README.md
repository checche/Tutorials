# Image Classification Tutorial

## 概要
- 本チュートリアルではPyTorchを用いて画像の分類を行う
- CNNの概要，PyTorchの使い方について学習することが目標
- 汎用的にコードを使いまわせるようにMNISTやCIFARのようなデータではなく，[Oxford-IIIT Pet Dataset - University of Oxford](https://www.robots.ox.ac.uk/~vgg/data/pets/)を用いる

## CNNとは
- CNNとはConvolutional Neural Networkの略称で，主に画像や自然言語処理の分野で用いられる
- CNNに関する詳しい説明は[この記事](https://jinbeizame.hateblo.jp/entry/kelpnet_cnn)を参照
- 畳み込みの計算は[この記事](http://deeplearning.stanford.edu/wiki/index.php/File:Convolution_schematic.gif)のgifを見るとイメージしやすい
- CNNに関する重要なトピックは「局所的受容野」と「重み共有」の2つ

### 局所的受容野
受容野とはあるニューロンへ入力を行う領域のことで，全結合層の場合，受容野は前の層の全ての出力となる．画像ではあるピクセルに着目したときその周辺のピクセルとは関連性が高いが，離れたピクセルとは関連性が低いことが考えられる．そのため受容野を局所的に絞ることで学習するパラメータ数を削減させつつ，関連するピクセル同士を効率的に演算できる仕組みが畳み込みである．  

### 重み共有
ある部分で需要な特徴(画像における線分やエッジなどの形状)は他の部分でも重要なことが多いため，畳み込み層では1枚のkernel(フィルタ)でスライドさせながら画像全体に対して畳み込みを行う．これにより全結合層と比較して学習するパラメータ数が圧倒的に少なくなり，学習や推論を高速に行える．  
このように入力データに対して同じパラメータを使い回すことを重み共有という．  

## PyTorchで画像分類

PyTorchで簡単なCNNを構築するには以下のような記述をする．  

```python
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

```

### 実際に学習を行う
