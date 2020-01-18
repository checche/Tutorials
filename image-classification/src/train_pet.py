import glob

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models import AlexNet

CLASS_NAMES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
    'horse', 'ship', 'truck'
)

BATCH_SIZE = 256
LR = 0.01
NUM_EPOCHS = 50

num_classes = len(CLASS_NAMES)


class PetDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_datasets(root):
    images = glob.glob(os.path.join(root, '*.jpg'))
    labels = []

    for img in images:
        # 犬は0, 猫は1のラベルをつける
        if img[0].islower():
            labels.append(0)
        else:
            labels.append(1)

    X_train, X_valid, y_train, y_valid = train_test_split(
        images,
        labels,
        shuffle=True,
        random_state=27
    )
    return X_train, X_valid, y_train, y_valid



def epoch_train(train_loader, model, optimizer, criterion, device=None):
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

    X_train, X_valid, y_train, y_valid = load_datasets('../images')
    dtrain = PetDataset(X_train, y_train)
    dvalid = PetDataset(X_valid, y_valid)

    train_loader = torch.utils.data.DataLoader(
        dtrain, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        dvalid, batch_size=BATCH_SIZE
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()
    model = AlexNet(num_classes=num_classes)
    model = torchvision.models.alexnet(pretrained=True, num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = epoch_train(train_loader, model, optimizer,
                                            criterion)
        test_loss, test_acc = epoch_eval(test_loader, model, criterion)

        print(f'EPOCH: [{epoch}/{NUM_EPOCHS}]')
        print(f'TRAIN LOSS: {train_loss:.3f}, TRAIN ACC: {train_acc:.3f}')
        print(f'TEST LOSS: {test_loss:.3f}, TEST ACC: {test_acc:.3f}')

        parameters = model.state_dict()
        torch.save(parameters, f'../weights/{epoch}.pth')
