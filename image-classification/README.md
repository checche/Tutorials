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

## PyTorchで画像分類を行う

### PyTorchでCNNを作成する

PyTorchで簡単なCNNを構築するには以下のような記述をする．  
今回は例としてAlexNetのようなものを実装する．  

```python
# image-classification/src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self, num_classes, in_channels=3):
        """ __init__メソッド内でモデルで用いる層を定義する """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """ forwardメソッドでは__init__で定義した層をどのように伝播させるのかを記述する """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = AlexNet(num_classes=2)
    inputs = torch.randn(4, 3, 32, 32)
    outputs = model(inputs)
    # (batch_size, num_classes)の(4, 2)サイズの出力が出る
    print(outputs.size())

```

### CIFAR10の学習を行う


```python
# image-classification/src/train_cifar.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# models.pyで定義したAlexNetをインポートする
from models import AlexNet

# CIFAR10の10クラスのクラス名
CLASS_NAMES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
    'horse', 'ship', 'truck'
)
# バッチサイズ，学習率，エポック数，クラス数を定義
BATCH_SIZE = 256
LR = 0.01
NUM_EPOCHS = 50
num_classes = len(CLASS_NAMES)


def epoch_train(train_loader, model, optimizer, criterion, device=None):
    """ モデルを1エポック学習させる関数 """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.train()
    model = model.to(device)
    epoch_loss = 0.
    total = 0
    correct = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / len(train_loader)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    return epoch_loss, accuracy


def epoch_eval(eval_loader, model, criterion, device=None):
    """ モデルの評価を1エポック分行う関数 """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    model = model.to(device)
    epoch_loss = 0.
    total = 0
    correct = 0
    for inputs, targets in eval_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        epoch_loss += loss.item() / len(eval_loader)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    return epoch_loss, accuracy


if __name__ == '__main__':
    # 画像のリサイズやaugumentationなどの変換処理を学習用とテスト用にそれぞれ定義
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR10のデータセットはtorchvisionを用いてこのように簡単に用意できる
    dtrain = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    # PyTorchではデータローダークラスを定義して，そこからデータを取り出す形で学習や評価を行う
    train_loader = torch.utils.data.DataLoader(
        dtrain, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    dtest = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        dtest, batch_size=BATCH_SIZE, shuffle=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 損失関数にクロスエントロピー誤差を用いる
    criterion = nn.CrossEntropyLoss()
    # モデルの定義
    model = AlexNet(num_classes=num_classes)
    # optimizerはSGD
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                weight_decay=5e-4)

    # 実際の学習部分
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = epoch_train(train_loader, model, optimizer,
                                            criterion)
        test_loss, test_acc = epoch_eval(test_loader, model, criterion)

        print(f'EPOCH: [{epoch}/{NUM_EPOCHS}]')
        print(f'TRAIN LOSS: {train_loss:.3f}, TRAIN ACC: {train_acc:.3f}')
        print(f'TEST LOSS: {test_loss:.3f}, TEST ACC: {test_acc:.3f}')

        # このように重みを保存する
        parameters = model.state_dict()
        torch.save(parameters, f'../weights/{epoch}.pth')

```


### 自作のデータセットクラスを定義する

- PyTorchではtorch.utils.data.Datasetというクラスを継承したデータセットクラスを作ることで，簡単に自作のデータセットを用いた学習が行える．
- 以下はdatasets.pyの中身

```python
import torch


class PetDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels):
        # 画像とクラス名(int)を受け取る
        self.images = images
        self.labels = labels

    def __len__(self):
        # __len__はデータセットのサイズを返すよう設定する
        return len(self.labels)

    def __getitem__(self, idx):
        # ここで各iterationの中でどのように画像トラベルのデータを取り出すかを記述する
        return self.images[idx], self.labels[idx]
```

### 自作のデータセットで学習を行う

- まずは[Oxford-IIIT Pet Dataset - University of Oxford](https://www.robots.ox.ac.uk/~vgg/data/pets/)からimages.tar.gzをダウンロードする
- それを解凍したimagesディレクトリをTutorials/image-classificationディレクトリに置く
- train_pet.pyの中身で実際に学習を行なっている

```python

```

